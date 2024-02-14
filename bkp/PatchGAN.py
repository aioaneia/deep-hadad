
import torch
import torch.nn as nn

from torch.nn                     import InstanceNorm2d, Sigmoid, Conv2d, ReLU, Module, AdaptiveAvgPool2d, Sequential, ConvTranspose2d, LeakyReLU, Linear, ModuleList, Dropout
from torch.nn.utils.spectral_norm import spectral_norm
from torch.nn.init                import kaiming_normal_, xavier_normal_, orthogonal_

from attention_mechanisms.SelfAttention import SelfAttention


class SqueezeExcitation(Module):
    """
        Squeeze and Excitation block.
        The block consists of a global average pooling layer followed by two convolutional layers.
        The first convolutional layer reduces the number of channels by a factor of 16.
        The second convolutional layer increases the number of channels back to the original number.
        The output of the second convolutional layer is passed through a sigmoid activation.
        The sigmoid output is multiplied with the input to the block.
    """
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcitation, self).__init__()

        # Ensure that reduction is not greater than in_channels
        if in_channels < reduction:
            reduction = in_channels
            print(f"Warning: reduction is greater than in_channels. Setting reduction to {in_channels}.")
        elif in_channels > reduction:
            print(f"Warning: in_channels is not divisible by reduction. Setting reduction to {in_channels}.")

        self.se = nn.Sequential(
            AdaptiveAvgPool2d(1),
            Conv2d(in_channels, in_channels // reduction, 1),
            ReLU(inplace=True),
            Conv2d(in_channels // reduction, in_channels, 1),
            Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class DHadadDiscriminator(Module):
    """
        Discriminator model based on PatchGAN.
        The model consists of 5 convolutional layers with instance normalization and leaky ReLU activation.
        The model also uses self-attention and squeeze-excitation blocks.
    """
    def __init__(self, in_channels, filter_sizes=[32, 64, 96, 128, 256, 384, 512], dropout_prob=0.2):
        super(DHadadDiscriminator, self).__init__()

        self.layers = ModuleList()

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
        self.self_attention_128  = SelfAttention(filter_sizes[3])  # 128
        self.self_attention_512 = SelfAttention(filter_sizes[6])  # 512

        self.dropout = Dropout(dropout_prob)


    # Convolutional block
    def conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, instance_norm=True, activation=LeakyReLU(0.2, inplace=False), use_sq_ex=True):
        """
            Convolutional block with instance normalization and leaky ReLU activation.
        """
        layers = [spectral_norm(Conv2d(in_channels, out_channels, kernel_size, stride, padding))]
        
        if instance_norm:
            layers.append(InstanceNorm2d(out_channels))
        
        if activation:
            layers.append(activation)
        
        layers.append(SqueezeExcitation(out_channels))

        return Sequential(*layers)

    # Forward pass
    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)

            if idx > 2:
                x = self.dropout(x)
        
            #print(f"Layer {idx}: {x.shape}")

            # Apply self-attention after the second layer
            if layer == self.layers[3]:
                x = self.self_attention_128(x)
                #print(f"Self-Attention: {x.shape}")
            
            # Apply self-attention after the third layer
            if layer == self.layers[6]:
                x = self.self_attention_512(x)
                #print(f"Self-Attention: {x.shape}")
        
        x = self.final_conv(x)

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