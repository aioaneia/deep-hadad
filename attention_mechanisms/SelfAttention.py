import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Module, Conv2d


####################################################################################################
# Self-Attention Layer
####################################################################################################
class SelfAttention(Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        # Define the key, query, and value convolution layers
        self.query_conv = Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = Conv2d(in_channels, in_channels, 1)

        # Scale factor to ensure stable gradients, as suggested by the Attention is All You Need paper
        # s elf.scale = torch.sqrt(torch.FloatTensor([in_channels // 8]))
        # Scale factor to ensure stable gradients
        # self.scale = (in_channels // 8) ** -0.5

        # Gamma parameter for learnable interpolation between input and attention
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # Flatten the spatial dimensions and compute query, key, value
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        # Compute attention and apply softmax
        attention = torch.bmm(query, key)  # / self.scale
        attention = F.softmax(attention, dim=-1)

        # Apply attention to the value
        out = torch.bmm(value, attention.permute(0, 2, 1))

        # Reshape the output and apply gamma
        out = out.view(batch_size, channels, width, height)

        # Learnable interpolation between input and attention output
        out = self.gamma * out + x

        return out


class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        # Pointwise convolution to compress the channel dimension.
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable parameter to scale the attention weights.

        self.softmax = nn.Softmax(dim=-1)  # Softmax to calculate attention weights.

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        energy = torch.bmm(proj_query, proj_key)  # Transpose check
        attention = self.softmax(energy)  # BX (N) X (N)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        return out


# class ChannelAttention(nn.Module):
#     """
#         Channel attention module.
#     """
#
#     def __init__(self, in_channels, ratio=16):
#         super(ChannelAttention, self).__init__()
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         original_x = x
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         out = self.sigmoid(out)
#
#         # Adding residual connection
#         return original_x + (original_x * out)
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         original_x = x
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#
#         # Adding residual connection
#         return original_x + (original_x * x)
#
#
# class CBAM(nn.Module):
#     """
#         CBAM module for self-attention.
#     """
#
#     def __init__(self, in_planes, ratio=16, kernel_size=7):
#         super(CBAM, self).__init__()
#         self.ca = ChannelAttention(in_planes, ratio)
#         self.sa = SpatialAttention(kernel_size)
#
#     def forward(self, x):
#         x = self.ca(x) * x
#         x = self.sa(x) * x
#         return x

# class SelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SelfAttention, self).__init__()
#
#         # Define the key, query, and value convolution layers
#         self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
#         self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
#         self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
#
#         # Scale factor to ensure stable gradients, as suggested by the Attention is All You Need paper
#         # s elf.scale = torch.sqrt(torch.FloatTensor([in_channels // 8]))
#         # Scale factor to ensure stable gradients
#         # self.scale = (in_channels // 8) ** -0.5
#
#         # Gamma parameter for learnable interpolation between input and attention
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         batch_size, channels, width, height = x.size()
#
#         # Flatten the spatial dimensions and compute query, key, value
#         query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
#         key = self.key_conv(x).view(batch_size, -1, width * height)
#         value = self.value_conv(x).view(batch_size, -1, width * height)
#
#         # Compute attention and apply softmax
#         attention = torch.bmm(query, key)  # / self.scale
#         attention = F.softmax(attention, dim=-1)
#
#         # Apply attention to the value
#         out = torch.bmm(value, attention.permute(0, 2, 1))
#
#         # Reshape the output and apply gamma
#         out = out.view(batch_size, channels, width, height)
#
#         # Learnable interpolation between input and attention output
#         out = self.gamma * out + x
#
#         return out