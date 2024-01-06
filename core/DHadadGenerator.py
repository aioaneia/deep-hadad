
import torch
import torch.nn as nn

from torch.nn                     import InstanceNorm2d, Sigmoid, Conv2d, ReLU, Dropout, Module, Sequential, ConvTranspose2d, LeakyReLU, Linear, ModuleList
from torch.nn.utils.spectral_norm import spectral_norm
from torch.nn.init                import kaiming_normal_, xavier_normal_, orthogonal_

from attention_mechanisms.MultiHeadAttention import MultiHeadAttention
from attention_mechanisms.SelfAttention      import SelfAttention
from attention_mechanisms.SpatialAttention   import SpatialAttention
from attention_mechanisms.ChannelAttention   import ChannelAttention

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


# Residual Block
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

    
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()

        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x

        return x


class DHadadGenerator(Module):
    """
        Generator model based on an adapted U-Net architecture.
        The model consists of an encoder and decoder with skip connections.
        The encoder consists of 8 convolutional layers with residual connections starting from the third layer.
        The decoder consists of 8 deconvolutional layers with skip connections.
        The model also uses self-attention and multi-head attention layers.
    """
    def __init__(self, in_channels, out_channels, 
        filter_sizes=[32, 64, 96, 128, 256, 384, 512, 1024], num_residual_blocks = 1, dropout_prob=0.2):
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
        self.encoders = ModuleList()
        
        # Residual blocks for encoder
        self.res_blocks_enc = ModuleList()
        
        # CBAM modules for encoder
        self.cbam_blocks_enc = ModuleList()  

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
    def conv_block(self, in_channels, out_channels, batch_norm=True, activation=LeakyReLU(0.2, inplace=False)):
        layers = [spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))]
        
        if batch_norm:
            layers.append(InstanceNorm2d(out_channels))
        
        layers.append(activation)

        return Sequential(*layers)

    # Deconvolutional block
    def deconv_block(self, in_channels, out_channels, batch_norm=True, activation=LeakyReLU(0.2, inplace=False)):
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

            # # Apply multi-head attention at a suitable position
            # if idx == 6:  # 512    
            #     # Reshape x for MultiHeadAttention
            #     batch_size, channels, height, width = x.shape
            #     # Reshaping to [batch_size, seq_len, feature_dim]
            #     x = x.view(batch_size, height * width, channels)

            #     # Apply MultiHeadAttention
            #     x = self.multi_head_attention(x)

            #     # Reshape back to original dimensions if needed
            #     x = x.view(batch_size, channels, height, width)

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
        if isinstance(m, Conv2d) or isinstance(m, ConvTranspose2d):
            # He initialization for layers followed by ReLU or LeakyReLU
            kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            #orthogonal_(m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, Linear):
            # Xavier initialization for fully connected layers
            xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)