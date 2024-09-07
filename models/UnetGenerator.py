import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_


class ResidualBlock(nn.Module):
    def __init__(self, num_filters, depth, total_depth, norm_layer=nn.BatchNorm2d, leaky_relu_slope=0.2, use_bias=True,
                 alpha_min=0.5):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(num_filters),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(num_filters)
        )

        self.alpha_l = self.calculate_alpha_l(depth, total_depth, alpha_min)

    def calculate_alpha_l(self, depth, total_depth, alpha_min):
        delta_alpha = (1 - alpha_min) / total_depth
        return 1 - delta_alpha * depth

    def forward(self, x):
        return self.alpha_l * x + self.conv_block(x)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        y = self.avg_pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet.
                For example, # if |num_downs| == 7, image of size 512x512 will become of size 4x4 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """

        super(UnetGenerator, self).__init__()

        # Construct the U-Net structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 16, ngf * 16,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
            total_depth=num_downs
        )

        # Add intermediate layers with ngf * 16 filters
        for i in range(num_downs - 7):
            unet_block = UnetSkipConnectionBlock(ngf * 16, ngf * 16, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout, total_depth=num_downs, depth=i+1)

        # Gradually reduce the number of filters
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 16, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, total_depth=num_downs, depth=num_downs-6)
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, total_depth=num_downs, depth=num_downs-5)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, total_depth=num_downs, depth=num_downs-4)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, total_depth=num_downs, depth=num_downs-3)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, total_depth=num_downs, depth=num_downs-2)

        # add the outermost layer
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer, total_depth=num_downs, depth=num_downs-1)

    def forward(self, input):
        """Standard forward"""

        return self.model(input)

    def initialize_weights(self, m, leaky_relu_slope=0.2):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


class UnetSkipConnectionBlock(nn.Module):
    """
        Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc,
                 inner_nc,
                 input_nc=None,
                 submodule=None,
                 outermost=False,
                 innermost=False,
                 norm_layer=nn.InstanceNorm2d,
                 use_dropout=False,
                 num_residual_blocks=3,
                 num_dense_layers=4,
                 growth_rate=32,
                 total_depth=7,
                 depth=0):
        super(UnetSkipConnectionBlock, self).__init__()

        self.outermost = outermost

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        # Downsampling
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)

        # Upsampling
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        # Add Squeeze and Excitation blocks
        se_block_inner = SEBlock(inner_nc)
        se_block_outer = SEBlock(outer_nc)

        # Add residual blocks
        residual_blocks_inner = nn.Sequential(*[ResidualBlock(inner_nc, depth + 1, total_depth, norm_layer=norm_layer, use_bias=use_bias)
                                                for _ in range(num_residual_blocks)])

        residual_blocks_outer = nn.Sequential(*[ResidualBlock(outer_nc, depth + 1, total_depth, norm_layer=norm_layer, use_bias=use_bias)
                                                for _ in range(num_residual_blocks)])

        if outermost:
            # outermost layer of the network
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            model = [downconv] + [submodule] + [uprelu, upconv, nn.Sigmoid()]
        elif innermost:
            # innermost layer of the network
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model = (
                    [downrelu, downconv] +
                    # [residual_blocks_inner, se_block_inner] +
                    [uprelu, upconv, upnorm]
            )
        else:
            # intermediate layer of the network
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)

            down = [
                downrelu, downconv, downnorm,
                residual_blocks_inner,
                se_block_inner
            ]

            up = [
                uprelu, upconv, upnorm,
                residual_blocks_outer,
                se_block_outer
            ]

            model = down + [submodule] + up

            if use_dropout:
                model += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Forward function (with skip connections)"""

        if self.outermost:
            return self.model(x)
        else:
            model_output = self.model(x)

            # Check if the feature maps from encoder and decoder are compatible for concatenation
            if x.shape[1] != model_output.shape[1] or x.shape[2] != model_output.shape[2]:
                raise ValueError(f"Feature maps are not compatible for concatenation. "
                                 f"Encoder output shape: {x.shape}, Decoder output shape: {model_output.shape}")

            # Concatenate the input with the output of the submodule
            x_cat = torch.cat([x, model_output], 1)

            return x_cat
