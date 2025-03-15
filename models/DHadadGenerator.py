import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torchvision.ops import DeformConv2d


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_dim, in_dim//8, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(in_dim//8, in_dim, 1)),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        attention_map = self.conv(x)
        return x + self.gamma * (x * attention_map)


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, (7, 1), padding=(3, 0)),        # Vertical strokes
            nn.Conv2d(1, 1, (1, 7), padding=(0, 3)),  # Horizontal strokes
            # nn.Conv2d(channels, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x) * x
        sa = self.spatial_attention(ca)
        return ca * sa


class SimplifiedSPADE(nn.Module):
    def __init__(self, norm_nc):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=8, num_channels=norm_nc, affine=False)
        self.attention = CBAM(norm_nc)

    def forward(self, x):
        return self.attention(self.norm(x))


class ResBlock(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # Main path convolutions
        self.conv_0 = spectral_norm(nn.Conv2d(fin, fmiddle, 3, padding=1))

        # Parallel dilated convolutions
        self.conv_dil1 = spectral_norm(nn.Conv2d(fmiddle, fmiddle//2, 3, padding=1, dilation=1))
        self.conv_dil2 = spectral_norm(nn.Conv2d(fmiddle, fmiddle//2, 3, padding=2, dilation=2))
        self.conv_dil4 = spectral_norm(nn.Conv2d(fmiddle, fmiddle//2, 3, padding=4, dilation=4))
        self.conv_dil8 = spectral_norm(nn.Conv2d(fmiddle, fmiddle//2, 3, padding=8, dilation=8))

        # Deformable convolution
        # self.offset_conv = spectral_norm(nn.Conv2d(fmiddle, 2*3*3, 3, padding=1))
        # self.conv_deform = DeformConv2d(fmiddle, fmiddle, 3, padding=1)

        # Final projection
        self.conv_out = spectral_norm(nn.Conv2d(fmiddle * 2, fout, 3, padding=1))

        # Shortcut
        if self.learned_shortcut:
            self.conv_shortcut = spectral_norm(nn.Conv2d(fin, fout, 1, bias=False))

        # Normalization
        self.norm_0 = SimplifiedSPADE(fin)
        self.norm_1 = SimplifiedSPADE(fmiddle)
        if self.learned_shortcut:
            self.norm_shortcut = SimplifiedSPADE(fin)

    def forward(self, x):
        # Shortcut path
        x_short = self.shortcut(x)

        # Main path
        dx = self.conv_0(self.actvn(self.norm_0(x)))
        dx = self.actvn(self.norm_1(dx))

        x1 = self.conv_dil1(dx)
        x2 = self.conv_dil2(dx)
        x4 = self.conv_dil4(dx)
        x8 = self.conv_dil8(dx)

        # Concatenate multi-scale features
        combined = torch.cat([x1, x2, x4, x8], dim=1)

        # Deformable convolution
        # offset = self.offset_conv(dx)
        # dx = self.conv_deform(dx, offset)

        # Final projection
        out = self.conv_out(combined)

        return x_short + out

    def shortcut(self, x):
        if self.learned_shortcut:
            return self.conv_shortcut(self.norm_shortcut(x))
        return x

    def actvn(self, x):
        return F.leaky_relu(x, 0.2)


class ProgressiveUpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1))
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        x = self.norm(x)
        return self.act(x)


class HybridUpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pixel_shuffle = ProgressiveUpSampling(in_channels, out_channels)
        self.transposed_conv = spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1))

    def forward(self, x):
        x1 = self.pixel_shuffle(x)
        x2 = self.transposed_conv(x)
        return (x1 + x2) / 2  # Average for stability


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.2)
        )
        self.res = nn.Conv2d(in_ch, out_ch, 1, stride=2) if in_ch != out_ch else None

    def forward(self, x):
        return self.conv(x) + (self.res(x) if self.res else x)


class DHadadGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9):
        super().__init__()
        # Initial convolution layer
        self.initial = nn.Sequential(
            nn.ZeroPad2d(3),
            spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=7)),
            nn.GroupNorm(8, ngf),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # --- Downsampling ---
        self.down_layers      = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        self.skip_weights     = nn.ParameterList()

        for i in range(n_downsampling):
            # Calculate input and output channels
            down_in_channels = ngf * (2 ** i)
            down_out_channels = ngf * (2 ** (i+1))

            self.down_layers.append(DownsampleBlock(down_in_channels, down_out_channels))

            self.skip_connections.append(nn.Sequential(
                nn.Conv2d(down_in_channels, down_in_channels, kernel_size=1),
                nn.GroupNorm(8, down_in_channels),
            ))
            self.skip_weights.append(nn.Parameter(torch.ones(1) * 0.5))

        # Add Self-Attention layer after downsampling
        self.attention_after_down = SelfAttention(ngf * (2 ** n_downsampling))

        # --- Multi-Scale ResBlocks ---
        self.res_blocks = nn.ModuleList()
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            block = ResBlock(ngf * mult, ngf * mult)
            self.res_blocks.append(block)

            # Add attention in the middle of ResBlocks
            if i == n_blocks // 2:
                self.attention_mid = SelfAttention(ngf * mult)

        # --- Upsampling with Attention Fusion ---
        self.up_layers = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            up_in_channels = ngf * mult if i == 0 else ngf * mult * 2
            up_out_channels = int(ngf * mult / 2)
            self.up_layers.append(ProgressiveUpSampling(up_in_channels, up_out_channels))

        self.reduce_channels = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(8, ngf),
            nn.LeakyReLU(0.2, True)
        )

        # Final output layer
        self.final = nn.Sequential(
            nn.ZeroPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Sigmoid()
        )

        # Initialize weights
        self.apply(initialize_weights)

        nn.init.constant_(self.attention_after_down.gamma, 0.3)
        nn.init.constant_(self.attention_mid.gamma, 0.3)

        # Special initialization for the final convolutional layer
        nn.init.xavier_normal_(self.final[1].weight, gain=0.01)
        nn.init.constant_(self.final[1].bias, 0.0)

    def forward(self, input):
        x = self.initial(input)

        # Downsampling
        skips = []
        for i, (down, skip_conv, skip_weight) in enumerate(zip(self.down_layers, self.skip_connections, self.skip_weights)):
            skips.append(skip_weight * skip_conv(x))
            x = down(x)

        # Self-Attention layer
        x = self.attention_after_down(x)

        # Resblocks layers
        for i, res in enumerate(self.res_blocks):
            x = res(x)
            if i == len(self.res_blocks) // 2:
                x = self.attention_mid(x)

        # Upsampling layers
        for i, (up, skip) in enumerate(zip(self.up_layers, reversed(skips))):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = x.contiguous()

        x = self.reduce_channels(x)

        return self.final(x)


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        if 'final' in str(m):
            nn.init.xavier_normal_(m.weight, gain=0.02)
        else:
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)

        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

