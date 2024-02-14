
import torch
import torch.nn            as nn
import torch.nn.functional as F

from pytorch_msssim import SSIM
from torchvision    import transforms

import torchvision.models as models

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        # Define Sobel filter for horizontal and vertical edge detection
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        # Sobel filter weights for x and y direction
        sobel_x_weights = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y_weights = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)

        self.sobel_x.weight = nn.Parameter(sobel_x_weights, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_y_weights, requires_grad=False)

    def forward(self, input, target):
        # Ensure input and target are in correct format
        if input.dim() != 4 or target.dim() != 4:
            raise ValueError("Expected input and target to be 4-dimensional BxCxHxW")

        # Apply Sobel filter to input and target images
        edge_input_x = self.sobel_x(input)
        edge_input_y = self.sobel_y(input)
        edge_target_x = self.sobel_x(target)
        edge_target_y = self.sobel_y(target)

        # Calculate edge magnitude for input and target
        edge_input_mag = torch.sqrt(edge_input_x ** 2 + edge_input_y ** 2)
        edge_target_mag = torch.sqrt(edge_target_x ** 2 + edge_target_y ** 2)

        # Calculate loss as Mean Squared Error between edge magnitudes of input and target
        loss = F.mse_loss(edge_input_mag, edge_target_mag)

        return loss

class GeometricConsistencyLoss(nn.Module):
    """
        Maintains geometric integrity of depth information.
        This will help in preserving the contours and shapes of letters in the displacement maps.
    """

    def __init__(self):
        super(GeometricConsistencyLoss, self).__init__()

    def forward(self, predicted_map, target_map):
        grad_x_pred, grad_y_pred     = self.compute_gradients(predicted_map)
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


class SharpnessLoss(nn.Module):
    """
        SharpnessLoss 
    """
    def __init__(self):
        super(SharpnessLoss, self).__init__()

        self.kernel = torch.tensor([[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, input, target):
        self.kernel  = self.kernel.to(input.device)
        sharp_input  = F.conv2d(input, self.kernel, padding=1)
        sharp_target = F.conv2d(target, self.kernel, padding=1)
        
        return F.mse_loss(sharp_input, sharp_target)


class DHadadLossFunctions:
    """
    Contains the loss functions used in the DHadad model
    """
    
    @staticmethod
    def l1_loss(input, target):
        """
        Calculates the L1 loss for a batch of images

        :param input: The input images
        :param target: The target images
        :return: The L1 loss for the batch
        """
        return nn.L1Loss()(input, target)


    @staticmethod
    def ssim_loss(input, target, data_range=1, size_average=True, channel=1):
        """
        Calculates the SSIM loss for a batch of images

        :param input: The input images
        :param target: The target images
        :return: The SSIM loss for the batch
        """

        ssim      = SSIM(data_range = data_range, size_average=size_average, channel=channel)
        ssim_loss = ssim(input, target)
        ssim_loss = 1 - ssim_loss

        return ssim_loss
    

    @staticmethod
    def adversarial_loss(predictions, labels):
        """
            Calculates adversarial loss for the generator and discriminator using the binary cross entropy with logits loss function 
            and the predictions and the labels.
        """

        return F.binary_cross_entropy_with_logits(predictions, labels)
    

    @staticmethod
    def depth_consistency_loss(input, target, epsilon=1e-6):
        """
        # Charbonnier Loss: sqrt((x - y)^2 + epsilon)

        :param input: The input images
        :param target: The target images
        :return: The depth consistency loss for the batch
        """

        loss = torch.mean(torch.sqrt((input - target) ** 2 + epsilon))

        return loss

    @staticmethod
    def geometric_consistency_loss(input, target):
        """
        Calculates the geometric consistency loss for a batch of images

        :param input: The input images
        :param target: The target images
        :return: The geometric consistency loss for the batch
        """
        return GeometricConsistencyLoss()(input, target)
    
    @staticmethod
    def sharpness_loss(input, target):
        """
        Calculates the sharpness loss for a batch of images
        
        Encourages spatial smoothness in the generated images.
        This can help in reducing noise and artifacts in the generated images.s.

        :param input: The input images
        :param target: The target images
        :return: The sharpness loss for the batch
        """
        return SharpnessLoss()(input, target)
    
    @staticmethod
    def edge_loss(input, target):
        """
        Calculates the edge loss for a batch of images
        
        Encourages spatial smoothness in the generated images.
        This can help in reducing noise and artifacts in the generated images.

        Pros:
            It's a good loss function for binary classification problems.
        Cons:
            It can be detrimental for sharp engravings.

        :param input: The input images
        :param target: The target images
        :return: The edge loss for the batch
        """
        return EdgeLoss()(input, target)


    def __init__(self):
        pass


    def compute_gradient_penalty(self,discriminator, damaged_dm, fake_samples, real_samples, lambda_gp=10.0):
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

