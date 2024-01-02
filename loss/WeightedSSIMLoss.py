
import torch.nn as nn

from pytorch_msssim import SSIM

class WeightedSSIMLoss(nn.Module):
    """
    A custom loss function that calculates a weighted Structural Similarity Index Measure (SSIM) loss.
    
    SSIM is used to assess the perceptual difference between two images, capturing textural and stylistic features,
    luminance, and contrast. This loss function weights the SSIM value to adjust its impact.
    
    Args:
        data_range (float): The data range of the input images. Default is 255 (8-bit images).
        size_average (bool): If True, the SSIM loss is averaged over the batch. Default is True.
        channel (int): Number of channels in the input images. Default is 1.
        weight (float): Weighting factor for the SSIM loss. Default is 1.0.
    """

    def __init__(self, data_range = 1, size_average = True, channel = 1, weight = 1.0):
        super(WeightedSSIMLoss, self).__init__()
        self.ssim   = SSIM(data_range=data_range, size_average=size_average, channel=channel)
        self.weight = weight

    def forward(self, img1, img2):
        """
        Calculates the weighted SSIM loss between two batches of images.

        Args:
            img1 (torch.Tensor): A batch of images.
            img2 (torch.Tensor): A batch of images to compare against img1.

        Returns:
            torch.Tensor: The weighted SSIM loss.
        """
        # Ensure input tensors are on the same device and have the same shape
        if img1.device != img2.device:
            raise ValueError("Input tensors are not on the same device.")
        if img1.shape != img2.shape:
            raise ValueError("Input tensors do not have the same shape.")

        ssim_loss = self.ssim(img1, img2)
        
        weighted_ssim_loss = self.weight * (1 - ssim_loss)

        return weighted_ssim_loss