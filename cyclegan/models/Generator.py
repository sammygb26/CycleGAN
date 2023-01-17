import torch
from torch import nn
from .ResidualBlock import ResidualBlock


class Generator(nn.Module):
    def __init__(self, nc_in: int, nc_out: int,
                 ngf: int = 64,
                 n_down: int = 2,
                 n_res: int = 6,
                 weights_path: str = None):
        super(Generator, self).__init__()

        self.nc_in: int = nc_in
        self.nc_out: int = nc_out
        self.ngf: int = ngf
        self.n_down: int = n_down
        self.n_res: int = n_res

        # Reflectional layer
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(nc_in, ngf, 7, 1, 0),
        ]

        # Instance-norm
        model += [
            nn.InstanceNorm2d(ngf)
        ]

        # Down-sampling
        for i in range(n_down):
            scale = 2 ** i
            model += [
                nn.Conv2d(ngf * scale, ngf * scale * 2, 3, 2, 1)
            ]

        # Residuals
        ngf_core = ngf * (2 ** n_down)
        for i in range(n_res):
            model += [
                ResidualBlock(ngf_core)
            ]

        # Up-sampling
        for i in range(n_down):
            scale = 2 ** (n_down - i)
            model += [
                nn.ConvTranspose2d(scale * ngf, scale * ngf // 2, 3, 2, 1, 1)
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, nc_out, 7, 1, 0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

        if weights_path:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        return self.model(x)
