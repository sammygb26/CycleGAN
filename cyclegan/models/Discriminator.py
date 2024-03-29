import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, nc_in: int, ndf: int, weights_path: (None|str) = None):
        super(Discriminator, self).__init__()

        self.nc_in: int = nc_in
        self.ndf: int = ndf

        self.conv_stack = nn.Sequential(
            nn.Conv2d(nc_in, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 8, 1, 4, 1, 1),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

        if weights_path:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        return self.conv_stack(x)

    def _init_weights(self, module : nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=0.02)
