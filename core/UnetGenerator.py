import torch
import torch.nn as nn
import functools

from torch.nn.init import kaiming_normal_, xavier_normal_, orthogonal_

from attention_mechanisms.SpatialAttention   import SpatialAttention
from attention_mechanisms.ChannelAttention   import ChannelAttention
from attention_mechanisms.SelfAttention      import SelfAttention


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
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original_x = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out     = avg_out + max_out
        out = self.sigmoid(out)

        # Adding residual connection
        return original_x + (original_x * out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'

        padding = 3 if kernel_size == 7 else 1

        self.conv1   = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original_x = x
        avg_out    = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x          = torch.cat([avg_out, max_out], dim=1)
        x          = self.conv1(x)

        # Adding residual connection
        return original_x + (original_x * x)


class CBAM(nn.Module):
    """
        CBAM module for self-attention.
    """
    def __init__(self, in_planes, ratio = 16, kernel_size = 7):
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

    def __init__(self, input_nc, output_nc, num_downs, ngf = 256, norm_layer = nn.InstanceNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. 
                For example, # if |num_downs| == 7, image of size 512x512 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """

        super(UnetGenerator, self).__init__()

        # construct unet structure 
        
        # add the innermost layer 
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule = None, norm_layer = norm_layer, innermost = True)  
        
        # add intermediate layers with ngf * 8 filters
        for i in range(num_downs - 5):          
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc = None, submodule=unet_block, norm_layer=norm_layer, use_dropout = use_dropout)
        
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2,     input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        
        # add the outermost layer 
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  


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
                 num_residual_blocks = 1):
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
        uprelu   = nn.ReLU(True)
        upnorm   = norm_layer(outer_nc)

        # Add attention blocks
        cbam_block_inner = CBAM(inner_nc, kernel_size=7, ratio=8)
        cbam_block_outer = CBAM(outer_nc, kernel_size=7, ratio=8)

        # Add SE blocks
        se_block_inner = SEBlock(inner_nc)
        se_block_outer = SEBlock(outer_nc)

        # Add residual blocks
        residual_blocks_inner = nn.Sequential(*[ResidualBlock(inner_nc, norm_layer=norm_layer, use_bias=use_bias) 
                                                for _ in range(num_residual_blocks)])
        residual_blocks_outer = nn.Sequential(*[ResidualBlock(outer_nc, norm_layer=norm_layer, use_bias=use_bias) 
                                                for _ in range(num_residual_blocks)])

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size = 4, stride = 2, padding = 1)
            model  = [downconv] + [submodule] + [uprelu, upconv, nn.Sigmoid()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size = 4, stride = 2, padding = 1, bias=use_bias)
            model  = [downrelu, downconv] + [residual_blocks_inner] + [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size = 4, stride = 2, padding = 1, bias=use_bias)
            down   = [downrelu, downconv, downnorm] + [cbam_block_inner] + [se_block_inner] + [residual_blocks_inner]
            up     = [uprelu, upconv, upnorm]       + [cbam_block_outer] + [se_block_outer] + [residual_blocks_outer]
            model  = down + [submodule] + up

            if use_dropout:
                model += [nn.Dropout(0.2)]

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