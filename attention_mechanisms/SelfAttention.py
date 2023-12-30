
import torch
import torch.nn            as nn
import torch.nn.functional as F

from torch.nn import Conv2d

####################################################################################################
# Self-Attention Layer
####################################################################################################
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        # Define the key, query, and value convolution layers
        self.query_conv = Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv   = Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = Conv2d(in_channels, in_channels, 1)

        # Check if CUDA (GPU) is available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Check if MPS (Multi-Process Service) is available
        # if torch.backends.mps.is_available():
        #     self.device = torch.device("mps")

        #     print("MPS device found.")
        # else:
        #     print("MPS device not found.")

        # Scale factor to ensure stable gradients, as suggested by the Attention is All You Need paper
        self.scale = torch.sqrt(torch.FloatTensor([in_channels // 8])).to(self.device)

        # Gamma parameter for learnable interpolation between input and attention
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # Flatten the spatial dimensions and compute query, key, value
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key   = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        self.scale = self.scale.to(self.device)

        # Compute attention and apply softmax
        attention = torch.bmm(query, key) / self.scale
        attention = F.softmax(attention, dim=-1)

        # Apply attention to the value
        out = torch.bmm(value, attention.permute(0, 2, 1))

        # Reshape the output and apply gamma
        out = out.view(batch_size, channels, width, height)

        # Learnable interpolation between input and attention output
        out = self.gamma * out + x

        return out