import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
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


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
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


class SpadeGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, label_nc=1):
        super().__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.res_blocks = nn.ModuleList()

        # Downsampling
        for i in range(n_downsampling):
            mult = 2 ** i
            self.down_layers.append(nn.Sequential(
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ))

        # Add Self-Attention layer after downsampling
        self.attention = SelfAttention(ngf * (2 ** n_downsampling))

        # Resblocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            self.res_blocks.append(SPADEResBlock(ngf * mult, ngf * mult, label_nc))

        # Upsampling layers
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)

            self.up_layers.append(nn.Sequential(
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ))

        # Final output layer
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Sigmoid()
        )

    def forward(self, input, segmap):
        x = self.initial(input)

        # Downsampling
        for down in self.down_layers:
            x = down(x)

        # Self-Attention
        x = self.attention(x)

        # Resblocks
        for res in self.res_blocks:
            x = res(x, segmap)

        # Upsampling
        for up in self.up_layers:
            x = up(x)

        x = self.final(x)

        return x
