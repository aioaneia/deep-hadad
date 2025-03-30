import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class SelfAttentionModule(nn.Module):
    """Self-attention module"""
    def __init__(self, in_dim):
        super(SelfAttentionModule, self).__init__()
        self.query_conv = spectral_norm(nn.Conv2d(in_dim, in_dim // 8, 1))
        self.key_conv = spectral_norm(nn.Conv2d(in_dim, in_dim // 8, 1))
        self.value_conv = spectral_norm(nn.Conv2d(in_dim, in_dim, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """Forward pass of the self-attention module"""
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out


class MinibatchStdDev(nn.Module):
    """Minibatch standard deviation layer"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Forward pass of the minibatch standard deviation layer"""
        batch_size, _, h, w = x.shape
        std = torch.std(x, dim=0, unbiased=False)  # [C, H, W]
        mean_std = torch.mean(std).expand(batch_size, 1, h, w)  # [B,1,H,W]
        return torch.cat([x, mean_std], dim=1)


class DiscriminatorBlock(nn.Module):
    """Discriminator block"""
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_filters, out_filters, 4, 2, 1)),
            nn.GroupNorm(8, out_filters),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(out_filters, out_filters, 3, padding=1)),
            nn.LeakyReLU(0.2, True)
        )
        self.res = nn.Sequential(
            nn.AvgPool2d(2, 2),
            spectral_norm(nn.Conv2d(in_filters, out_filters, 1))
        )

    def forward(self, x):
        """Forward pass of the discriminator block"""
        return self.block(x) + self.res(x)


class DeepHadadDiscriminator(nn.Module):
    """DeepHadadDiscriminator"""
    def __init__(self, input_nc, ndf=64, n_downsample=3):
        super(DeepHadadDiscriminator, self).__init__()

        self.model = self.create_discriminator(input_nc, ndf, n_downsample)

        self.apply(initialize_weights)

        self.init_self_attention()


    @staticmethod
    def create_discriminator(input_nc, ndf, n_downsample):
        """Create the discriminator"""
        layers = []
        in_filters = input_nc

        for i in range(n_downsample):
            out_filters = ndf * min(2**i, 8)

            layers.append(
                DiscriminatorBlock(in_filters, out_filters)
            )

            if i == 2 or i == 3:
                layers.append(SelfAttentionModule(out_filters))
            else:
                layers.append(nn.Identity())

            in_filters = out_filters

        # Output layer
        layers.append(MinibatchStdDev())
        layers.append(spectral_norm(nn.Conv2d(in_filters + 1, in_filters, 3, padding=1)))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(spectral_norm(nn.Conv2d(in_filters, 1, 3, padding=1)))
        return nn.Sequential(*layers)

    def init_self_attention(self):
        """Initialize the self-attention module"""
        for module in self.modules():
            if isinstance(module, SelfAttentionModule):
                nn.init.constant_(module.gamma, 0.1)

    def forward(self, input):
        """Forward pass of the discriminator"""
        return self.model(input)


def initialize_weights(m):
    """Initialize the weights of the discriminator"""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.InstanceNorm2d, nn.GroupNorm, nn.LayerNorm)):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
