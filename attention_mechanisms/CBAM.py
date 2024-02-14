
from attention_mechanisms.SpatialAttention import SpatialAttention
from attention_mechanisms.ChannelAttention import ChannelAttention
from torch.nn import Module


class CBAM(Module):
    """
        CBAM module for self-attention.
    """

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()

        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x

        return x
