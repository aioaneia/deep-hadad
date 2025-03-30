"""This file contains the loss functions used in the DHadad model."""
import torch
import torch.nn as nn
from pytorch_msssim import SSIM


class DHadadLossFunctions:
    """Contains the loss functions used in the DHadad model"""
    def __init__(self, device):
        self.device = device
        self.ssim = SSIM(data_range=1.0, channel=1).to(device)

        # Sobel kernels for edge detection
        self.sobel_kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        self.sobel_kernel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)

        # Create conv layers for Sobel filters
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        
        # Set the weights
        self.conv_x.weight = nn.Parameter(self.sobel_kernel_x, requires_grad=False)
        self.conv_y.weight = nn.Parameter(self.sobel_kernel_y, requires_grad=False)

    @staticmethod
    def l1_loss(pred, target):
        """L1 loss"""
        return nn.functional.l1_loss(pred, target)

    def ssim_loss(self, pred, target):
        """SSIM loss"""
        return 1 - self.ssim(pred, target)

    def sobel_edges(self, x):
        """Sobel edges"""
        # Get kernels on same device as input
        self.conv_x.to(x.device)
        self.conv_y.to(x.device)
        sobel_x = self.conv_x(x)
        sobel_y = self.conv_y(x)

        return torch.sqrt(sobel_x.pow(2) + sobel_y.pow(2) + 1e-8)

    def edge_loss(self, pred, target):
        """Edge loss"""
        target_edges = self.sobel_edges(target)
        pred_edges = self.sobel_edges(pred)

        # Balanced edge importance (scale=2.5)
        edge_importance = torch.sigmoid(target_edges * 2.5)

        edge_diff = torch.abs(target_edges - pred_edges)
        weighted_edge_diff = edge_diff * (1.0 + 2.0 * edge_importance)

        return weighted_edge_diff.mean()

    @staticmethod
    def depth_continuity_loss(pred, target):
        """
        Enforces smoothness and continuity in depth maps to avoid abrupt height changes
        that would be unlikely in natural glyph erosion.
        """
        # Calculate gradient in x and y directions for predicted output
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        # Calculate gradient in x and y directions for target
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        # Get gradient differences
        diff_dx = pred_dx - target_dx
        diff_dy = pred_dy - target_dy

        # Apply L1 loss
        diff_dx = torch.abs(diff_dx)
        diff_dy = torch.abs(diff_dy)

        # Weighting based on target gradients - higher penalties for deviating from
        # significant height transitions in the target
        # weights_dx = torch.exp(torch.abs(target_dx) * 5.0)
        # weights_dy = torch.exp(torch.abs(target_dy) * 5.0)

        weights_dx = 1.0 + torch.abs(target_dx) * 2.0
        weights_dy = 1.0 + torch.abs(target_dy) * 2.0

        # Apply weights to differences
        weighted_diff_dx = diff_dx * weights_dx
        weighted_diff_dy = diff_dy * weights_dy

        # Compute final loss
        loss = (weighted_diff_dx.sum() + weighted_diff_dy.sum()) / (weights_dx.sum() + weights_dy.sum() + 1e-8)

        return loss

    @staticmethod
    def adversarial_loss(real_pred, fake_pred, loss_type='hinge'):
        """Stabilized adversarial loss"""
        if loss_type == 'hinge':
            gen_loss = -torch.mean(fake_pred)
            real_loss = nn.functional.relu(1 - real_pred).mean()
            fake_loss = nn.functional.relu(1 + fake_pred).mean()
            return gen_loss, real_loss + fake_loss
        else:  # WGAN-GP
            return -torch.mean(fake_pred), torch.mean(fake_pred) - torch.mean(real_pred)

    @staticmethod
    def compute_gradient_penalty(discriminator, damaged_dm, fake_samples, real_samples):
        """Compute gradient penalty"""
        batch_size = real_samples.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=real_samples.device)
        epsilon = epsilon.expand_as(real_samples)
        interpolates = (epsilon * real_samples + (1 - epsilon) * fake_samples).requires_grad_(True)
        interpolated_input = torch.cat([damaged_dm.expand_as(interpolates), interpolates], dim=1)
        d_interpolates = discriminator(interpolated_input)
        ones = torch.ones_like(d_interpolates, requires_grad=False)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()
        return gradient_penalty
