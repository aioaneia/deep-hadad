import math

import torch
import torch.nn            as nn
import torch.nn.functional as F

from torch.nn                     import InstanceNorm2d, Sigmoid, Conv2d, ReLU, Dropout, Module, AdaptiveAvgPool2d, Sequential, ConvTranspose2d, LeakyReLU
from torch.nn.utils.spectral_norm import spectral_norm
from torch.nn.init                import kaiming_normal_, xavier_normal_, orthogonal_

#Alternative Attention Mechanisms

# Convolutional Block Attention Module (CBAM):

# This module sequentially infers attention maps along two dimensions (channel and spatial), which could enhance the model's focus on relevant features in depth maps.


# Squeeze-and-Excitation (SE) Blocks:

# Already used in your discriminator, these blocks could also be beneficial in the generator. They perform dynamic channel-wise feature recalibration, potentially enhancing the model's ability to focus on important features.
# Spatial Attention:

# Similar to CBAM, but focuses solely on spatial features. This could be particularly useful if the spatial arrangement of features in your depth maps is a significant aspect of the reconstruction task.
# Local Attention:

# Local attention mechanisms focus on small neighborhoods, which might be more computationally efficient than global attention mechanisms like multi-head attention. They can be particularly useful for tasks like depth map reconstruction where local features are important.
# Layer Attention:

# Instead of applying attention within a layer, layer attention mechanisms aggregate information across different layers. This can be particularly effective in U-Net architectures where features from different scales are combined.


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


####################################################################################################
# Residual Block
####################################################################################################
class ResidualBlock(Module):
    """
        Residual Block with instance normalization.
    """
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            InstanceNorm2d(in_channels),
            ReLU(inplace=True),
            Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = ReLU()
        self.fc2   = Conv2d(in_channels // ratio, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out     = avg_out + max_out

        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'

        padding = 3 if kernel_size == 7 else 1

        self.conv1   = Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out    = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x          = torch.cat([avg_out, max_out], dim=1)
        x          = self.conv1(x)

        return self.sigmoid(x)
    
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()

        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x

        return x

####################################################################################################
# Deep Hadad Generator Model based on an adapted U-Net architecture
# 
####################################################################################################
class DHadadGenerator(Module):
    """
        Generator model based on an adapted U-Net architecture.
        The model consists of an encoder and decoder with skip connections.
        The encoder consists of 8 convolutional layers with residual connections starting from the third layer.
        The decoder consists of 8 deconvolutional layers with skip connections.
        The model also uses self-attention and multi-head attention layers.
    """
    def __init__(self, in_channels, out_channels, filter_sizes=[32, 64, 96, 128, 256, 384, 512, 1024],
        self_attention_levels=[2], num_residual_blocks = 1, dropout_prob=0.1):
        """
            in_channels: Number of input channels
            out_channels: Number of output channels
            filter_sizes: List of filter sizes for each layer
            self_attention_levels: List of encoder levels where self-attention is applied
            num_residual_blocks: Number of residual blocks in the encoder and decoder
            dropout_prob: Dropout probability
        """
        super(DHadadGenerator, self).__init__()
        
        # Initialize encoder
        self.initialize_encoder(in_channels, filter_sizes, num_residual_blocks)
        
        # Decoder 
        self.initialize_decoder(out_channels, filter_sizes, num_residual_blocks)

        # Initialize the weights using He initialization
        self.apply(self.initialize_weights)

        # Dropout layer
        self.dropout = Dropout(dropout_prob)

        # Add Self-Attention layers
        self.self_attention_128  = SelfAttention(filter_sizes[3]) # 128
        self.self_attention_256  = SelfAttention(filter_sizes[4]) # 256
        self.self_attention_512  = SelfAttention(filter_sizes[6]) # 512
        self.self_attention_1024 = SelfAttention(filter_sizes[7]) # 1024

        # Initialize Multi-Head Attention
        #self.multi_head_attention = MultiHeadAttention(in_channels=512, num_heads=8, dropout=dropout_prob)

    # Initialize encoder
    def initialize_encoder(self, in_channels, filter_sizes, num_residual_blocks):
        # Encoder blocks
        self.encoders = nn.ModuleList()
        
        # Residual blocks for encoder
        self.res_blocks_enc = nn.ModuleList()

        # CBAM modules for encoder
        self.cbam_blocks_enc = nn.ModuleList()  

        previous_channels = in_channels
        
        for idx, filter_size in enumerate(filter_sizes):
            # First encoder layer does not use batch normalization
            if idx == 0:
                self.encoders.append(self.conv_block(previous_channels, filter_size, batch_norm=False))
            else:
                self.encoders.append(self.conv_block(previous_channels, filter_size))

            if idx > 2:
                # Adding residual blocks for encoder
                for _ in range(num_residual_blocks):
                    self.res_blocks_enc.append(ResidualBlock(filter_size))
    
                # Add CBAM module after each encoder layer
                self.cbam_blocks_enc.append(CBAM(filter_size))

            previous_channels = filter_size

    # Initialize decoder
    def initialize_decoder(self, out_channels, filter_sizes, num_residual_blocks):
        self.decoders        = nn.ModuleList()
        self.res_blocks_dec  = nn.ModuleList()
        self.cbam_blocks_dec = nn.ModuleList()

        reversed_filter_sized = list(reversed(filter_sizes))

        for idx, filter_size in enumerate(reversed_filter_sized):
            if idx == 0:
                # First decoder layer does not use batch normalization
                self.decoders.append(self.deconv_block(filter_size, filter_size // 2))
                self.res_blocks_dec.append(ResidualBlock(filter_size // 2))
                self.cbam_blocks_dec.append(CBAM(filter_size // 2))
            elif idx == len(filter_sizes) - 1:
                # Last decoder layer does not use batch normalization and uses sigmoid activation
                self.decoders.append(self.deconv_block(filter_size * 2, out_channels, batch_norm=False, activation=Sigmoid()))
            else:
                # Second to last decoder layer does not use batch normalization
                self.decoders.append(self.deconv_block(filter_size * 2, reversed_filter_sized[idx + 1]))
                # Adding residual blocks for decoder
                self.res_blocks_dec.append(ResidualBlock(reversed_filter_sized[idx + 1]))
                # Add CBAM module
                self.cbam_blocks_dec.append(CBAM(reversed_filter_sized[idx + 1]))

    # Convolutional block
    def conv_block(self, in_channels, out_channels, batch_norm=True, activation=nn.LeakyReLU(0.2, inplace=False)):
        layers = [spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))]
        
        if batch_norm:
            layers.append(InstanceNorm2d(out_channels))
        
        layers.append(activation)

        return Sequential(*layers)

    # Deconvolutional block
    def deconv_block(self, in_channels, out_channels, batch_norm=True, activation=nn.LeakyReLU(0.2, inplace=False)):
        layers = [ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        
        if batch_norm:
            layers.append(InstanceNorm2d(out_channels))
        
        layers.append(activation)

        return Sequential(*layers)

    # Forward pass
    def forward(self, x):
        # Encoder with skip connections
        skip_connections = []

        for idx, encoder in enumerate(self.encoders):
            x = encoder(x)
            
            if idx > 2:
                # Apply residual blocks
                x = self.res_blocks_enc[idx - 3](x)
                
                # Apply CBAM after each encoder layer
                x = self.cbam_blocks_enc[idx - 3](x)

            # Skip connection except for the last layer
            if idx < len(self.encoders) - 1:
                skip_connections.append(x)
            
            # Apply self-attention
            if idx == 3:
                x = self.self_attention_128(x)
            elif idx == 4:
                x = self.self_attention_256(x)
            elif idx == 6:
                x = self.self_attention_512(x)

        # Decoder with skip connections
        for idx, decoder in enumerate(self.decoders):
            x = decoder(self.dropout(x) if idx < 2 else x)

            # Apply CBAM and residual blocks except for the last layer
            if idx < len(self.decoders) - 1:
                x = self.res_blocks_dec[idx](x)
                x = self.cbam_blocks_dec[idx](x)

            # Skip connection except for the last layer
            if idx < len(self.decoders) - 1:
                skip_connection = skip_connections[-(idx + 1)]
                
                x = torch.cat([x, skip_connection], dim=1)

            if idx == 0: # 1024
                x = self.self_attention_1024(x)
            elif idx == 2: # 512
                x = self.self_attention_512(x)
            
        return x

    # Function to initialize weights
    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # If ReLU activation follows
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.Linear):
            # For fully connected layers
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.ConvTranspose2d):
            orthogonal_(m.weight)

        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


####################################################################################################
# Squeeze Excitation Block
####################################################################################################
class SqueezeExcitation(Module):
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcitation, self).__init__()

        # Ensure that reduction is not greater than in_channels
        if in_channels < reduction:
            reduction = in_channels
        
        self.se = nn.Sequential(
            AdaptiveAvgPool2d(1),
            Conv2d(in_channels, in_channels // reduction, 1),
            ReLU(inplace=True),
            Conv2d(in_channels // reduction, in_channels, 1),
            Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

####################################################################################################
# Deep Hadad Discriminator based on PatchGAN
####################################################################################################
class DHadadDiscriminator(Module):
    """
        Discriminator model based on PatchGAN.
        The model consists of 5 convolutional layers with instance normalization and leaky ReLU activation.
        The model also uses self-attention and squeeze-excitation blocks.
    """
    def __init__(self, in_channels, filter_sizes=[32, 64, 96, 128, 256, 384, 512, 1024], use_spectral_norm=True):
        super(DHadadDiscriminator, self).__init__()

        self.layers = nn.ModuleList()

        previous_channels = in_channels

        for idx, filter_size in enumerate(filter_sizes):
            stride = 1 if idx == 0 or idx == (len(filter_sizes) - 1) else 2

            self.layers.append(self.conv_block(previous_channels, filter_size, stride=stride))

            previous_channels = filter_size

        # Final Convolution to output single channel prediction
        self.final_conv = self.conv_block(filter_sizes[-1], 1, kernel_size=4, stride=1, instance_norm=False, activation=Sigmoid(), use_sq_ex=False)

        # Initialize the weights using He initialization
        self.apply(self.initialize_weights)

        # Adding Self-Attention
        self.self_attention_96  = SelfAttention(filter_sizes[2])  # 96
        self.self_attention_512 = SelfAttention(filter_sizes[6])  # 512
        
    # Convolutional block
    def conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, instance_norm=True, activation=LeakyReLU(0.2, inplace=False), use_sq_ex=True):
        """
            Convolutional block with instance normalization and leaky ReLU activation.
        """
        layers = [spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))]
        
        if instance_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        
        if activation:
            layers.append(activation)
        
        layers.append(SqueezeExcitation(out_channels))

        return Sequential(*layers)

    # Forward pass
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

            # Apply self-attention after the second layer
            # if layer == self.layers[2]:
            #     x = self.self_attention_96(x)
            
            # Apply self-attention after the third layer
            if layer == self.layers[6]:
                x = self.self_attention_512(x)

        return self.final_conv(x)
    
    # Function to initialize weights
    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # If ReLU activation follows
            kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.Linear):
            # For fully connected layers
            xavier_normal_(m.weight)
        elif isinstance(m, nn.ConvTranspose2d):
            orthogonal_(m.weight)

        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
