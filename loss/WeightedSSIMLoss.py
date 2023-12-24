import os
import time
import sys
import numpy as np
import pandas as pd
import logging
import cv2

import torch.nn as nn

from pytorch_msssim import SSIM

########################################################################################
# Perceptual Loss
# Captures textural and stylistic features.
# luminance, and contrast in an image.
# A higher weight is crucial as it emphasizes on the perceptual similarity, which is key for letter reconstruction.
########################################################################################
class WeightedSSIMLoss(nn.Module):
    def __init__(self, data_range=255, size_average=True, channel=1, weight=1.0):
        super(WeightedSSIMLoss, self).__init__()

        self.ssim   = SSIM(data_range=data_range, size_average=size_average, channel=channel)
        self.weight = weight

    def forward(self, img1, img2):

        ssim_loss          = self.ssim(img1, img2)
        weighted_ssim_loss = self.weight * (1 - ssim_loss)

        return weighted_ssim_loss