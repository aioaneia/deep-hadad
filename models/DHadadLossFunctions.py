import torch
import torch.nn as nn
import torch.nn.functional as F

from lpips import LPIPS
from pytorch_msssim import SSIM


class GeometricConsistencyLoss(nn.Module):
    """
        Maintains geometric integrity of depth information.
        This will help in preserving the contours and shapes of letters in the displacement maps.
    """

    def __init__(self):
        super(GeometricConsistencyLoss, self).__init__()

    def forward(self, predicted_map, target_map):
        grad_x_pred, grad_y_pred = self.compute_gradients(predicted_map)
        grad_x_target, grad_y_target = self.compute_gradients(target_map)

        # Calculate the loss as the mean squared error between the gradients of the predicted and target maps
        loss_x = F.mse_loss(grad_x_pred, grad_x_target)
        loss_y = F.mse_loss(grad_y_pred, grad_y_target)

        # Combine the losses
        loss = loss_x + loss_y

        return loss

    def compute_gradients(self, map):
        # Function to compute gradients in the x and y direction
        grad_x = map[:, :, :, :-1] - map[:, :, :, 1:]
        grad_y = map[:, :, :-1, :] - map[:, :, 1:, :]

        return grad_x, grad_y


class DHadadLossFunctions:
    """
    Contains the loss functions used in the DHadad model
    """

    def __init__(self, device):
        self.device = device

        self.lpips_alex = LPIPS(net='alex').to(self.device)

    def l1_loss(self, input, target):
        """
        Calculates the L1 loss for a batch of images

        :param input: The input images
        :param target: The target images
        :return: The L1 loss for the batch
        """

        return nn.L1Loss()(input, target)

    def ssim_loss(self, input, target, data_range=1, size_average=True, channel=1):
        """
        Calculates the SSIM loss for a batch of images

        :param input: The input images
        :param target: The target images
        :return: The SSIM loss for the batch
        """

        ssim = SSIM(data_range=data_range, size_average=size_average, channel=channel)
        ssim_loss = ssim(input, target)
        ssim_loss = 1 - ssim_loss

        return ssim_loss

    def lpips_loss(self, input, target):
        """
        Calculates the LPIPS loss for a batch of images

        :param input: The input images
        :param target: The target images
        :return: The LPIPS loss for the batch
        """
        input = input.to(self.device)
        target = target.to(self.device)

        input = (input - input.min()) / (input.max() - input.min() + 1e-8)
        target = (target - target.min()) / (target.max() - target.min() + 1e-8)

        loss = self.lpips_alex(input, target).mean()

        return loss

    def adversarial_loss(self, predictions, labels):
        """
            Calculates adversarial loss for the generator and discriminator using the binary cross entropy with logits loss function 
            and the predictions and the labels.
        """

        return F.binary_cross_entropy_with_logits(predictions, labels)

    def geometric_consistency_loss(self, input, target):
        """
        Calculates the geometric consistency loss for a batch of images

        :param input: The input images
        :param target: The target images
        :return: The geometric consistency loss for the batch
        """
        return GeometricConsistencyLoss()(input, target)

    def compute_gradient_penalty(self, discriminator, damaged_dm, fake_samples, real_samples, lambda_gp=10.0):
        """
        Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
        The gradient penalty enforces the Lipschitz constraint for the critic (discriminator).
        """
        batch_size = real_samples.size(0)

        # Generate random epsilon for the interpolation
        epsilon = torch.rand(batch_size, 1, 1, 1, device=real_samples.device)
        epsilon = epsilon.expand_as(real_samples)

        # Interpolate between real and fake samples
        interpolates = epsilon * real_samples + (1 - epsilon) * fake_samples
        interpolates = interpolates.requires_grad_(True)

        # Concatenate the damaged_dm with the interpolates along the channel dimension
        # Make sure damaged_dm is expanded to the same batch size as interpolates if necessary
        interpolated_input = torch.cat([damaged_dm.expand_as(interpolates), interpolates], dim=1)

        # Pass the interpolated input through the discriminator
        d_interpolates = discriminator(interpolated_input)

        # Create a tensor of ones that is the same size as d_interpolates output,
        # which will be used to compute gradients
        ones = torch.ones_like(d_interpolates, requires_grad=False)

        # Compute gradients of d_interpolates with respect to interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Reshape gradients to calculate norm per batch sample
        # Gradients has shape (batch_size, num_channels, H, W)
        # Flatten the gradients such that each row contains all the gradients for one sample
        gradients = gradients.view(batch_size, -1)

        # Calculate the norm of the gradients for each sample (2-norm across each row)
        # Add a small epsilon for numerical stability
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Calculate gradient penalty as the mean squared distance to 1 of the gradients' norms
        gradient_penalty = ((gradients_norm - 1) ** 2).mean() * lambda_gp

        return gradient_penalty
