import torch

from diffusion_model.utils import test as utils_test
from diffusion_model.networks import utils as networks_utils
from diffusion_model.networks import unet

def main():
    print('###########################')
    utils_test.hello_world()
    time_dim = 10
    batch_size = 2
    time = torch.rand(batch_size, 1)
    time_embed = networks_utils.SinusoidalPositionEmbeddings(dim=time_dim)
    embeddings = time_embed.forward(time)
    print(embeddings.shape)
    print(embeddings)

    net = unet.Unet(
        dim=32,
        channels=3,
        dim_mults=(1, 2, 4))
    print(net)


if __name__ == "__main__":
    main()