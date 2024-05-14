import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_

from attention_mechanisms.ChannelAttention import ChannelAttention
from attention_mechanisms.SelfAttention import SelfAttention
from attention_mechanisms.SpatialAttention import SpatialAttention


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        # Define the key, query, and value convolution layers
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)

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


class DenseBlock(nn.Module):
    def __init__(self, num_layers, input_channels, growth_rate, bn_size=4, drop_rate=0):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.layer_list = nn.ModuleList()

        for i in range(num_layers):
            layer = DenseLayer(input_channels + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.layer_list.append(layer)

    def forward(self, x):
        for layer in self.layer_list:
            new_features = layer(x)
            x = torch.cat([x, new_features], 1)  # Concatenate output features to input features.
        return x


class DenseLayer(nn.Sequential):
    def __init__(self, input_channels, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(input_channels)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(input_channels, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class ResidualBlock(nn.Module):
    def __init__(self, num_filters, norm_layer=nn.BatchNorm2d, leaky_relu_slope=0.2, use_bias=True):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(num_filters),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(num_filters),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class ChannelAttention(nn.Module):
    """
        Channel attention module.
    """

    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original_x = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)

        # Adding residual connection
        return original_x + (original_x * out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'

        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original_x = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        # Adding residual connection
        return original_x + (original_x * x)


class CBAM(nn.Module):
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

    def __init__(self, input_nc, output_nc, num_downs, ngf=128, norm_layer=nn.InstanceNorm2d, use_dropout=False):
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

        # construct unet structure 

        # add the innermost layer 
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)

        # add intermediate layers with ngf * 8 filters
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)

        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)

        # add the outermost layer 
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)

    def forward(self, input):
        """Standard forward"""

        return self.model(input)

    def initialize_weights(self, m, leaky_relu_slope=0.2):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=leaky_relu_slope)
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
                 num_residual_blocks=2):
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

        # Using bilinear upsampling to avoid checkerboard artifacts in transposed convolutions
        upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Add attention blocks
        cbam_block_inner = CBAM(inner_nc, kernel_size=7, ratio=8)
        cbam_block_inner_2 = CBAM(inner_nc * 2, kernel_size=7, ratio=8)

        # Add Squeeze and Excitation blocks
        se_block_inner = SEBlock(inner_nc)
        se_block_outer = SEBlock(outer_nc)

        # Add residual blocks
        residual_blocks_inner = nn.Sequential(*[ResidualBlock(inner_nc, norm_layer=norm_layer, use_bias=use_bias)
                                                for _ in range(num_residual_blocks)])

        residual_blocks_inner_2 = nn.Sequential(*[ResidualBlock(inner_nc * 2, norm_layer=norm_layer, use_bias=use_bias)
                                                for _ in range(num_residual_blocks)])

        residual_blocks_outer = nn.Sequential(*[ResidualBlock(outer_nc, norm_layer=norm_layer, use_bias=use_bias)
                                                for _ in range(num_residual_blocks)])
        if outermost:
            # outermost layer of the network
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            model = [downconv] + [submodule] + [uprelu, upconv, nn.Sigmoid()]
        elif innermost:
            # innermost layer of the network
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model = [downrelu, downconv] + [uprelu, upconv, upnorm]
        else:
            # intermediate layer of the network
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]  # cbam_block_inner, se_block_inner,
            up = [uprelu, upconv, upnorm]  # cbam_block_outer, se_block_outer,
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
