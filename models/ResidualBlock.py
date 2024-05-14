

from torch.nn import Sequential, InstanceNorm2d, Conv2d, ReLU, Module


class ResidualBlock(Module):
    """
        Residual Block with instance normalization.
    """

    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.block = Sequential(
            Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            InstanceNorm2d(in_channels),
            ReLU(inplace=True),
            Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)
