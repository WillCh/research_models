import math
import torch

import torch.nn.functional as F
from inspect import isfunction
from functools import partial
from torch import nn, einsum
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

# Function to either take the value if it's not None, otherwise, to use
# the default_value.
def defaultValue(value, default_value):
    if value is not None:
        return value
    return default_value() if isfunction(default_value) else default_value

# Residual net layer.
class Residual(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn

    # x is the tensor whose first dimension is batch size.
    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.activation_fn(x, *args, **kwargs) + x
    
# Conv upsampling layer, it upsamples the image by scale of 2 (twice).
def Upsample(dim_in: int, dim_out: int | None = None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        # The conv2d layer does not change the image size.
        nn.Conv2d(dim_in, defaultValue(dim_out, dim_in),
                   kernel_size=3, padding=1),
    )

# Conv downsampling layer, it downsamples the image by half.
def Downsample(dim_in: int, dim_out: int | None = None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        # Split each image into 4 smaller images, and stack
        # all four sub-images into the channels.
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        # Use the conv to sequeeze the dim as output one.
        nn.Conv2d(dim_in * 4, defaultValue(dim_out, dim_in), 1),
    )

# Positional embedding layer.
# We use the positional embedding to encode the timestamp t.
# Here the timestamp means the t in the diffusion process.
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    # The time has dim: [B, 1], here 1 mean the correpsonding timestamp
    # for that img. Output has dimesion as [B, dim].
    # Recall the sin position embeddings are:
    # PE(pos, 2i) = sin(pos/10000^{2i/d}) 
    # = sin(pos * exp (-1 * i * log(10000) / (d/2))).
    # PE(pos, 2i+1) = cos(pos/10000^{2i/d}).
    def forward(self, time: torch.Tensor):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# Assemble the CNN layers
class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x: torch.Tensor):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) / (var + eps).rsqrt()
        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


# This is a general CNN block which has group norm and SiLU.
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class BlockForConditionInfo(nn.Module):
    """This block consume the conditional embedding and inserts as separate channels."""

    # The dim is the input image's channel number.
    def __init__(self, dim: int, dim_out: int, conditional_info_emb_dim: int, groups=8):
        super().__init__()
        # Applys a MLP to encode the conditional info to be the same as input channel number.
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(conditional_info_emb_dim, dim))
        )
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, conditional_info_emb: torch.Tensor):
        conditional_info = self.mlp(conditional_info_emb)
        conditional_info = rearrange(conditional_info, "b c -> b c 1 1")
        x = x + conditional_info 
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim: int, dim_out: int, *, time_emb_dim=None,
                 conditional_info_emb_dim=None, groups=8):
        super().__init__()
        # Applys a MLP to encode the time embeddings (i.e. SinusoidalPosition).
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )
        
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        # The conditional info is added after block 2.
        self.conditional_info_inserter = (
            BlockForConditionInfo(dim_out, dim_out, conditional_info_emb_dim, groups)
            if conditional_info_emb_dim is not None
            else None
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None = None, 
                conditional_info: torch.Tensor | None = None):    
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            # Split the time embeddings into two tensors.
            # One tensor is added as shift and another one is multiple as scale.
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        if self.conditional_info_inserter is not None and conditional_info is not None:
            h = self.conditional_info_inserter(h, conditional_info)
        return h + self.res_conv(x)

# This layer uses CNN to encode/decode the input vectors.
# Idea is to treat channels as the embedding/token vector, each pixel as one token.
# conduct the self-attention among all pixels.
class Attention(nn.Module):
    # dim is the input tensor's channel dimensions.
    # dim_head is the Q/K/V's dimension for each head.
    def __init__(self, dim: int, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        # Note for conv, input is [B, C, H, W], and same for the output.
        self.to_qkv = nn.Conv2d(in_channels=dim, out_channels=hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(in_channels=hidden_dim, out_channels=dim, kernel_size=1)

    # We assume the input tensor has shape of [batch_size, channels, height, weight]
    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        # Use the CNN layer to encode the QKV into a big tensor, then
        # split it into 3 views (split by channel).
        qkv = self.to_qkv(x).chunk(3, dim=1)
        # Regroup the QKV tensors, here h means head.
        # After mapping, QKV has shape [batch, heads, channels, x*y]
        # thus, we only need to perform tranformer on the last dim.
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        # Q * K in the attention. Here means we perform attention along the channel
        # dimention, for each pixel (i means the pixel in Q, j means pixel in K).
        sim = einsum("b h c i, b h c j -> b h i j", q, k)
        # amax is argmax. Here we subtract the max dim to make softmax more numerical stable
        # here we don't want the amax's gradient pass to the params, thus, uses detach.
        # The detach just create a stopped gradient placerholder for that original tensor.
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h c j -> b h i c", attn, v)
        out = rearrange(out, "b h (x y) c -> b (h c) x y", x=h, y=w)
        return self.to_out(out)

# See https://arxiv.org/abs/1812.01243.
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

# Group norm layer. It's applied before transformer.
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)