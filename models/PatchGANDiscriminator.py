
import torch.nn as nn

from torch.nn import InstanceNorm2d, Sigmoid, Conv2d, Sequential, ConvTranspose2d, LeakyReLU, \
    Linear, Dropout
from torch.nn.utils.spectral_norm import spectral_norm
from torch.nn.init import kaiming_normal_, xavier_normal_


def initialize_weights(m):
    if isinstance(m, Conv2d):
        nn.init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


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


def conv_block(in_channels, out_channels,
               kernel_size=(5, 3),  # 5,3 for letter shapes
               stride=2,  # 2
               padding=1,  # 1
               instance_norm=True,
               activation_fn=LeakyReLU(0.2), deep_layer=False):
    if deep_layer:
        kernel_size = 3

    layers = [spectral_norm(Conv2d(in_channels, out_channels, kernel_size, stride, padding))]

    if instance_norm:
        layers.append(InstanceNorm2d(out_channels))

    if activation_fn:
        layers.append(activation_fn)

    return Sequential(*layers)


def build_discriminator(
        base_channels,
        filter_sizes=[64, 128, 256, 512, 512, 512, 1024],
        use_dropout=False,
        use_se_block=False):
    """
        Discriminator model based on PatchGAN.
    """

    layers = []

    # in_channels = base_channels
    in_channels = base_channels * 2

    # First layer without instance normalization
    layers.append(conv_block(in_channels, filter_sizes[0], stride=1, instance_norm=False))

    # Second layer with instance normalization
    layers.append(conv_block(filter_sizes[0], filter_sizes[1], stride=2))

    # Third layer with instance normalization
    layers.append(conv_block(filter_sizes[1], filter_sizes[2], stride=2))

    # Fourth layer with instance normalization
    layers.append(conv_block(filter_sizes[2], filter_sizes[3], stride=2))

    # Fifth layer with instance normalization
    layers.append(conv_block(filter_sizes[3], filter_sizes[4], stride=2))
    if use_dropout:
        layers.append(Dropout(0.2))

    # Sixth layer with instance normalization
    layers.append(conv_block(filter_sizes[4], filter_sizes[5], stride=2))

    if use_se_block:
        layers.append(SEBlock(filter_sizes[5]))

    if use_dropout:
        layers.append(Dropout(0.2))

    # Seventh layer with instance normalization
    layers.append(conv_block(filter_sizes[5], filter_sizes[6], stride=1))

    if use_dropout:
        layers.append(Dropout(0.2))

    # Final convolutional layer
    layers.append(conv_block(filter_sizes[6], 1, stride=1, instance_norm=False, activation_fn=Sigmoid()))

    # Create the sequential model
    model = Sequential(*layers)

    # Apply the weight initialization
    model.apply(initialize_weights)

    return model
