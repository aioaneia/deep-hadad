import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class SelfAttentionModule(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttentionModule, self).__init__()
        self.query_conv = spectral_norm(nn.Conv2d(in_dim, in_dim // 8, 1))
        self.key_conv = spectral_norm(nn.Conv2d(in_dim, in_dim // 8, 1))
        self.value_conv = spectral_norm(nn.Conv2d(in_dim, in_dim, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
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


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_features, in_features, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_features, in_features, 3, padding=1))
        )

    def forward(self, x):
        return x + self.block(x)


class MinibatchStdDev(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size, _, height, width = x.shape
        std = torch.std(x, dim=0, unbiased=False)
        mean_std = torch.mean(std)
        mean_std = mean_std.expand((batch_size, 1, height, width))
        return torch.cat([x, mean_std], dim=1)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_filters, out_filters, normalize=True):
        super(DiscriminatorBlock, self).__init__()

        layers = [spectral_norm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1))]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_filters))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class NewDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=4, use_sigmoid=False):
        super(NewDiscriminator, self).__init__()

        # self.multi_scale_discriminators = nn.ModuleList([
        #     self.create_single_discriminator(input_nc, ndf, n_layers, use_sigmoid)
        #     for _ in range(3)  # 3 scales
        # ])

        self.model = self.create_single_discriminator(input_nc, ndf, n_layers, use_sigmoid)

        # self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.apply(initialize_weights)

        self.init_self_attention()


    def create_single_discriminator(self, input_nc, ndf, n_layers, use_sigmoid):
        layers = []

        in_filters = input_nc

        for i in range(n_layers):
            out_filters = ndf * min(2**i, 8)

            layers.append(DiscriminatorBlock(in_filters, out_filters, normalize=(i > 0)))

            if i == 1 or i == 3:
                layers.append(SelfAttentionModule(out_filters))

            in_filters = out_filters

        # Output layer
        layers.append(ResidualBlock(in_filters))
        layers.append(MinibatchStdDev())
        layers.append(spectral_norm(nn.Conv2d(in_filters + 1, in_filters, 3, padding=1)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(spectral_norm(nn.Conv2d(in_filters, 1, 4, padding=0)))

        if use_sigmoid:
            layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def init_self_attention(self):
        for module in self.modules():
            if isinstance(module, SelfAttentionModule):
                nn.init.constant_(module.gamma, 0.0)

    def forward(self, input):
        return self.model(input)

    # def forward(self, input):
    #     results = []
    #     for i, d in enumerate(self.multi_scale_discriminators):
    #         if i != 0:
    #             input = self.downsample(input)
    #         results.append(d(input))
    #     return results


def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.InstanceNorm2d):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


