import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


class DepthEncoder(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=3, norm_layer=nn.BatchNorm2d, activation=nn.LeakyReLU(0.2, True)):
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 activation]

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      activation]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Start with small gamma

    def forward(self, x):
        batch_size, C, width, height = x.size()

        # Create query, key, and value projections
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # [B, N, C/8]
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  # [B, C/8, N]
        proj_value = self.value_conv(x).view(batch_size, C, width * height)  # [B, C, N]

        d_k = proj_key.size(1)
        epsilon = 1e-8  # Prevent division by zero

        # Compute energy (scaled dot-product attention)
        energy = torch.bmm(proj_query, proj_key) / (d_k ** 0.5 + epsilon)

        # Stabilize energy before softmax
        # energy = energy - torch.max(energy, dim=-1, keepdim=True)[0]
        energy = torch.clamp(energy, min=-5.0, max=5.0)

        # Apply softmax
        attention = F.softmax(energy, dim=-1)

        # Perform batch matrix multiplication between value and attention map
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

        # Reshape output to match the input dimensions
        out = out.view(batch_size, C, width, height)

        # Apply gamma scaling and residual connection
        out = self.gamma * out + x

        return out


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False) # Use BatchNorm instead of InstanceNorm
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


class SPADEResBlock(nn.Module):
    def __init__(self, fin, fout, label_nc):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)

        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)

    def forward(self, x, segmap):
        x_s = self.shortcut(x, segmap)
        dx = self.conv_0(self.actvn(self.norm_0(x, segmap)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, segmap)))
        out = x_s + dx
        return out

    def shortcut(self, x, segmap):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, segmap))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class ProgressiveUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.pixelshuffle = nn.PixelShuffle(2)
        self.norm = nn.BatchNorm2d(out_channels) # Use BatchNorm instead of InstanceNorm
        self.act = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pixelshuffle(x)
        x = self.norm(x)
        return self.act(x)


class BilinearUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)  # Use BatchNorm instead of InstanceNorm
        self.act = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class Spade2Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, label_nc=1):
        super().__init__()

        # Depth Encoder
        self.depth_encoder = DepthEncoder(input_nc, ngf, n_downsampling)

        # Initial convolution layer
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.BatchNorm2d(ngf),  # Use BatchNorm instead of InstanceNorm
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )

        self.down_layers      = nn.ModuleList()
        self.up_layers        = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        self.res_blocks       = nn.ModuleList()
        self.skip_weights     = nn.ParameterList()

        # Downsampling layers
        for i in range(n_downsampling):
            mult = 2 ** i
            dowm_in_channels = ngf * mult
            down_out_channels = ngf * mult * 2

            self.down_layers.append(nn.Sequential(
                nn.Conv2d(dowm_in_channels, down_out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(down_out_channels), # Use BatchNorm instead of InstanceNorm
                nn.LeakyReLU(0.2, True),
                nn.Dropout(0.3)  # Add dropout here
            ))

            self.skip_connections.append(nn.Sequential(
                nn.Conv2d(dowm_in_channels, dowm_in_channels, kernel_size=1),
                nn.InstanceNorm2d(dowm_in_channels)
            ))

            self.skip_weights.append(nn.Parameter(torch.ones(1) * 0.1))

        # Add Self-Attention layer after downsampling
        self.attention_after_down = SelfAttention(ngf * (2 ** n_downsampling))

        # Spade Resblocks layers
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            self.res_blocks.append(SPADEResBlock(ngf * mult, ngf * mult, label_nc))
            if i == n_blocks // 2:  # Add attention in the middle of ResBlocks
                self.attention_mid = SelfAttention(ngf * mult)
            # add dropout here

        # Upsampling layers
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            up_in_channels = ngf * mult if i == 0 else ngf * mult * 2
            up_out_channels = int(ngf * mult / 2)

            if i % 2 == 0:
                self.up_layers.append(ProgressiveUpsampling(up_in_channels, up_out_channels))
            else:
                self.up_layers.append(BilinearUpsampling(up_in_channels, up_out_channels))


        self.reduce_channels = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2, True)
        )

        # Final output layer
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Sigmoid()
        )

        # Initialize weights
        self.apply(initialize_weights)

        # Special initialization for SPADE blocks
        for block in self.res_blocks:
            nn.init.normal_(block.norm_0.mlp_gamma.weight, mean=0.0, std=0.01)
            nn.init.normal_(block.norm_0.mlp_beta.weight, mean=0.0, std=0.01)
            nn.init.normal_(block.norm_1.mlp_gamma.weight, mean=0.0, std=0.01)
            nn.init.normal_(block.norm_1.mlp_beta.weight, mean=0.0, std=0.01)

        nn.init.constant_(self.attention_after_down.gamma, 0.1)
        nn.init.constant_(self.attention_mid.gamma, 0.1)

        # Special initialization for the final convolutional layer
        nn.init.xavier_normal_(self.final[1].weight)
        nn.init.constant_(self.final[1].bias, 0.0)

    def forward(self, input, segmap):
        # Encode depth
        # x = self.depth_encoder(input)

        x = self.initial(input)

        # Downsampling
        skips = []
        for i, (down, skip_conv, skip_weight) in enumerate(zip(self.down_layers, self.skip_connections, self.skip_weights)):
            skips.append(skip_weight * skip_conv(x))
            x = down(x)

        # Self-Attention layer
        x = self.attention_after_down(x)

        # SPADE Resblocks layers
        for i, res in enumerate(self.res_blocks):
            x = res(x, segmap)
            if i == len(self.res_blocks) // 2:
                x = self.attention_mid(x)

        # Upsampling layers
        for i, (up, skip) in enumerate(zip(self.up_layers, reversed(skips))):
            x = up(x)

            if x.shape[1] != skip.shape[1]:
                print(f"Channel mismatch in upsampling: {x.shape[1]} != {skip.shape[1]}")

                # if x.shape[1] > skip.shape[1]:
                #     x = x[:, :skip.shape[1], :, :]  # Reduce channels if more than skip
                # elif x.shape[1] < skip.shape[1]:
                #     skip = skip[:, :x.shape[1], :, :]  # Reduce skip connection channels if more

                x = nn.Conv2d(x.shape[1], skip.shape[1], kernel_size=1, bias=False)(x)

            x = torch.cat([x, skip], dim=1)

        x = self.reduce_channels(x)

        x = self.final(x)

        return x


def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # Initialize Conv2D and ConvTranspose2D layers
        if m.weight is not None:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.InstanceNorm2d):
        # Initialize InstanceNorm2D layers
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.BatchNorm2d):
        # Initialize BatchNorm2D layers
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

