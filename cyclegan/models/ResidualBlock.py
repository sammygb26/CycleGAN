from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, nc: int):
        super(ResidualBlock, self).__init__()

        self.nc: int = nc
        self.block = nn.Sequential(
            nn.Conv2d(nc, nc, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(nc, nc, 3, 1, 1, bias=False)
        )

    def forward(self, x):
        fx = self.block(x)
        return fx + x
