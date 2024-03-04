import torch

import diffusion_model.networks.utils as networks_utils
from torch import nn, einsum
from functools import partial


# Define the U-net. The input to the Unet is a noise image batches,
# we expect it returns the noise. The input tensor shape is [B, channels, h, w].
class Unet(nn.Module):
    # dim: a general unit of the internal layer's dimenstion;
    #      e.g. the channel num of downsample/upsample cnn.
    # init_dim: there is a cnn as the first layer, it converts
    #      input image channels to this init_dim.
    # out_dim: the channels of final returned tensor. The U-net
    #      is mirror for upsampling and downsampling. Thus, the image
    #      size is the same as the noise size (w, h).
    # channels: the number of the image channels;
    # resnet_block_groups: the hyperparams for the group normalization
    #      in the resnet sub-module.
    def __init__(
        self,
        dim: int,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        conditional_embed_model: nn.Module | None = None,
        conditional_embed_size: int | None = None,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = networks_utils.defaultValue(init_dim, dim)
        # changed to 1 and 0 from 7,3
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)

        # The dims is [init_dim, dim, 2dim, 4dim, 8dim].
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # The in_out is [(init_dim, dim), (dim, 2dim), (2dim, 4dim), (4dim, 8dim)]
        in_out = list(zip(dims[:-1], dims[1:]))

        print('in-out')
        print(in_out)

        resnet_block = partial(
            networks_utils.ResnetBlock,
            conditional_info_emb_dim=conditional_embed_size,
            groups=resnet_block_groups)

        # Embedding heads for possible captions.
        self.conditional_embed_model = conditional_embed_model
        
        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            networks_utils.SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        # Build stacks to downsample the images.
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            print('dim_in: ' + str(dim_in) + '; dim_out: ' + str(dim_out))
            self.downs.append(
                nn.ModuleList(
                    [
                        resnet_block(dim_in, dim_in, time_emb_dim=time_dim),
                        resnet_block(dim_in, dim_in, time_emb_dim=time_dim),
                        # Apply residual connection for norm layer + linear attention.
                        networks_utils.Residual(
                            networks_utils.PreNorm(
                                dim_in, networks_utils.LinearAttention(dim_in))),
                        # Apply connv to downsample the image.
                        # If not the last layer, we split image into 4 sub images
                        # and stack them into channels to perform dim_out. Thus,
                        # image size shrink by half.
                        networks_utils.Downsample(dim_in, dim_out)
                        if not is_last
                        # For the last layer, we simply conduct a conv2d, without change
                        # the image size.
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        # The mid dim is 8 dim, the time dim is 4 dim.
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = networks_utils.Residual(
            networks_utils.PreNorm(mid_dim, networks_utils.Attention(mid_dim)))
        self.mid_block2 = resnet_block(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            print('dim_in: ' + str(dim_in) + '; dim_out: ' + str(dim_out))
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        resnet_block(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        resnet_block(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        networks_utils.Residual(
                            networks_utils.PreNorm(
                                dim_out, networks_utils.LinearAttention(dim_out))),
                        # Similar as downsample, the non-last layer upsample double
                        # the image size by nearest upsampling.
                        networks_utils.Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = networks_utils.defaultValue(out_dim, channels)

        self.final_res_block = resnet_block(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    # x is the input images with size as [B, channels, h, w];
    # time is the timestamp tensor (in diffusion steps),  with size as [B, 1]
    def forward(self, x: torch.Tensor, time: torch.Tensor, 
                conditional_info: torch.Tensor | None = None, x_self_cond=None):
        if self.self_condition:
            x_self_cond = networks_utils.defaultValue(
                x_self_cond, lambda: torch.zeros_like(x))
            # Stack the zeros or identiy images into the channels.
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        # The clone will allow the gradients flow back.
        # r will be directly stack to the last output layer as a residual connection.
        residual_input_from_x = x.clone()
        # Encode the conditional info.
        if self.conditional_embed_model is not None and conditional_info is not None:
            conditional_info_embed = self.conditional_embed_model(conditional_info)
        else:
            conditional_info_embed = None
        
        t = self.time_mlp(time)

        # we use h hold the output of downsample cnns.
        # Those output will be passed to upsampler as a residual connection.
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x=x, time_emb=t, conditional_info=conditional_info_embed)
            h.append(x)

            x = block2(x=x, time_emb=t, conditional_info=conditional_info_embed)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            # Here we attach the output of downsample's cnn layer,
            # it's similar as a residual connection.
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x=x, time_emb=t, conditional_info=conditional_info_embed)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x=x, time_emb=t, conditional_info=conditional_info_embed)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, residual_input_from_x), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)