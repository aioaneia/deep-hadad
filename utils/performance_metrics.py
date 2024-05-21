import cv2
import time

import torch
import torch.nn.functional as F
import torch.nn as nn

from math import log10
from pytorch_msssim import ssim


def compute_psnr(img1, img2, max_psnr=100):
    """
    Computes the Peak Signal to Noise Ratio (PSNR) between two images.

    References:
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    https://www.mathworks.com/help/images/ref/psnr.html
    https://www.mathworks.com/help/images/peak-signal-to-noise-ratio-psnr.html

    :param img1: The first image
    :param img2: The second image
    :param max_psnr: The maximum PSNR value
    :return: The PSNR between the two images
    """
    mse = F.mse_loss(img1, img2)

    if mse == 0:
        return max_psnr

    # Add a small positive number inside the square root 
    # to ensure the input is always non-negative
    return 20 * log10(1.0 / torch.sqrt(mse + 1e-10))


def compute_ssim(img1, img2):
    """
    Computes the Structural Similarity Index (SSIM) between two images.

    References:
    https://en.wikipedia.org/wiki/Structural_similarity
    https://en.wikipedia.org/wiki/SSIM_index

    :param img1: The first image
    :param img2: The second image
    :return: The SSIM between the two images
    """
    return ssim(img1, img2)


class SobelFilter(nn.Module):
    """
    SobelFilter is a simple Sobel filter that computes the edge magnitude of an image.
    """
    def __init__(self):
        super(SobelFilter, self).__init__()
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        # Sobel filter weights
        sobel_x_weights = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3,
                                                                                                                3)
        sobel_y_weights = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3,
                                                                                                                3)

        self.sobel_x.weight = nn.Parameter(sobel_x_weights, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_y_weights, requires_grad=False)

    def forward(self, x):
        # Assumes x is a single-channel image
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)

        return torch.sqrt(edge_x ** 2 + edge_y ** 2)


sobel = SobelFilter()


def compute_edge_similarity(img1, img2):
    """
    Computes the edge similarity index (ESI) between two images.
    :param img1: The first image
    :param img2: The second image
    :return: The ESI between the two images
    """
    sobel.to(img1.device)

    edge1 = sobel(img1)
    edge2 = sobel(img2)

    return ssim(edge1, edge2)


def combined_score(psnr, ssim, edge_similarity):
    """
    Computes a combined score that takes into account PSNR, SSIM, and ESI.
    """
    weighted_psnr = 0.4 * psnr
    weighted_ssim = 0.3 * ssim
    weighted_edge = 0.3 * edge_similarity

    total_score = weighted_psnr + weighted_ssim + weighted_edge

    return total_score


def print_performance_metrics(epoch, num_epochs, epoch_time, loss_weights,
                              avg_psnr, min_psnr, max_psnr, std_psnr,
                              avg_ssim, min_ssim, max_ssim, std_ssim,
                              avg_esi, min_esi, max_esi, std_esi,
                              avg_combined_score, performance_metrics,
                              gen_loss, loss_components, dis_loss, gen_optim, dis_optim):
    print("---------------------------------------------------------------------------------------------------")
    print(f" Epoch:                    {epoch}/{num_epochs}")
    print(f" Time:                     {epoch_time:.2f}s Current: {time.strftime('%H:%M:%S', time.gmtime())}")
    print(f" Epoch Average PSNR:       {avg_psnr:.4f} (Min: {min_psnr:.4f}, Max: {max_psnr:.4f}, Std: {std_psnr:.4f})")
    print(f" Epoch Average SSIM:       {avg_ssim:.4f} (Min: {min_ssim:.4f}, Max: {max_ssim:.4f}, Std: {std_ssim:.4f})")
    print(f" Epoch Average ESI:        {avg_esi:.4f}  (Min: {min_esi:.4f},  Max: {max_esi:.4f},  Std: {std_esi:.4f})")
    print(f" Epoch Combined Score:     {avg_combined_score:.4f}")
    print(f" Epoch Loss Weights:       {loss_weights.weights}")
    print(f" Epoch Generator Loss:     {gen_loss:.4f}")
    print(f" Epoch L1 Loss:            {loss_components[0].item():.4f}")
    print(f" Epoch SSIM Loss:          {loss_components[1].item():.4f}")
    print(f" Epoch LPIPS Loss:         {loss_components[2].item():.4f}")
    print(f" Epoch Adv Loss:           {loss_components[3].item():.4f}")
    print(f" Eposh Geom Loss:          {loss_components[4].item():.4f}")
    print(f" Epoch Discriminator Loss: {dis_loss:.4f}")
    print(f" Epoch Generator LR:       {gen_optim.param_groups[0]['lr']:.6f}")
    print(f" Epoch Discriminator LR:   {dis_optim.param_groups[0]['lr']:.6f}")
    print(f" Epoch Performance:        {performance_metrics}")
    print("---------------------------------------------------------------------------------------------------")
    print("")


########################################################################################
# Main
# Test the performance metrics
########################################################################################
if __name__ == "__main__":
    # Constants
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the images
    img1 = cv2.imread("test_images/ground_truth.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("test_images/restored.png", cv2.IMREAD_GRAYSCALE)

    # Calculate the PSNR
    psnr = compute_psnr(img1, img2)
    print(f"PSNR: {psnr}")

    # Calculate the SSIM
    ssim = compute_ssim(img1, img2)
    print(f"SSIM: {ssim}")

    # Calculate the ESI
    edge_similarity = compute_edge_similarity(img1, img2, device)
    print(f"Edge Similarity: {edge_similarity}")

    # Calculate the combined score
    score = combined_score(psnr, ssim, edge_similarity)
    print(f"Combined Score: {score}")
