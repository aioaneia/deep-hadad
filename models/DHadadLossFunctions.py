import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import SSIM
from pytorch_msssim import ms_ssim


class FrequencyDomainLoss(nn.Module):
    def __init__(self, alpha=1.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, y_true, y_pred):
        # Convert to frequency domain with epsilon to avoid instability
        fft_true = torch.fft.fft2(y_true + self.eps)
        fft_pred = torch.fft.fft2(y_pred + self.eps)

        # Compute magnitude spectrum, adding a small constant for stability
        mag_true = torch.abs(fft_true) + self.eps
        mag_pred = torch.abs(fft_pred) + self.eps

        # Compute log-magnitude spectrum
        log_mag_true = torch.log(mag_true)
        log_mag_pred = torch.log(mag_pred)

        # Compute MSE in log-magnitude spectrum
        mse_loss = F.mse_loss(log_mag_true, log_mag_pred)

        return self.alpha * mse_loss


class TVLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = max(1, x[:, :, 1:, :].numel())  # Avoid division by zero
        count_w = max(1, x[:, :, :, 1:].numel())  # Avoid division by zero
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]) + self.eps, 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]) + self.eps, 2).sum()
        return (h_tv / count_h + w_tv / count_w) / batch_size


# class MSSSIMPerceptualLoss(nn.Module):
#     def __init__(self, eps=1e-6):
#         super().__init__()
#         self.ms_ssim = ms_ssim
#         self.eps = eps  # Small epsilon for numerical stability
#
#     def forward(self, img1, img2):
#         # Ensure images are in [0, 1] range
#         img1 = torch.clamp(img1, 0.0, 1.0) + self.eps
#         img2 = torch.clamp(img2, 0.0, 1.0) + self.eps
#
#         # Compute MS-SSIM with epsilon to avoid instability
#         ms_ssim_value = self.ms_ssim(img1, img2, data_range=1.0, size_average=True)
#
#         # Ensure the final result doesn't produce NaN or Inf
#         if torch.isnan(ms_ssim_value).any() or torch.isinf(ms_ssim_value).any():
#             raise ValueError(f"NaN or Inf detected in MS-SSIM computation!")
#
#         return 1 - ms_ssim_value


class DHadadLossFunctions:
    """
    Contains the loss functions used in the DHadad model
    """

    def __init__(self, device):
        self.device       = device
        self.tv_loss      = TVLoss().to(device)
        self.ssim_loss    = SSIM(data_range=1.0, size_average=True, channel=1).to(device)
        # self.ms_ssim_loss = MSSSIMPerceptualLoss().to(device)
        self.freq_loss    = FrequencyDomainLoss().to(device)

    def l1_loss(self, y_true, y_pred):
        return F.l1_loss(y_true, y_pred)


    def ssim_loss(self, y_true, y_pred):
        return 1 - self.ssim_loss(y_true, y_pred)


    # def ms_ssim_loss(self, y_true, y_pred):
    #     return self.ms_ssim_loss(y_true, y_pred)


    def gradient_difference_loss(self, y_true, y_pred):
        def gradient(x):
            h, w = x.shape[2:]
            dx = x[:, :, 1:, :] - x[:, :, :h - 1, :]
            dy = x[:, :, :, 1:] - x[:, :, :, :w - 1]
            return dx, dy

        dx_true, dy_true = gradient(y_true)
        dx_pred, dy_pred = gradient(y_pred)

        # Ensure shapes match
        if dx_true.shape != dx_pred.shape:
            dx_pred = dx_pred[:, :, :, :dx_true.shape[3]]
        if dy_true.shape != dy_pred.shape:
            dy_pred = dy_pred[:, :, :dy_true.shape[2], :]

        dx_loss = torch.mean(torch.abs(dx_true - dx_pred))
        dy_loss = torch.mean(torch.abs(dy_true - dy_pred))

        return dx_loss + dy_loss


    def tv_loss(self, input):
        """
        Calculates the TV loss for a batch of images
        """
        return self.tv_loss(input)


    def frequency_domain_loss(self, y_true, y_pred):
        return self.freq_loss(y_true, y_pred)


    def hinge_loss_discriminator(self, real_pred, fake_pred):
        """
        Hinge loss for adversarial training of the discriminator
        """
        real_loss = torch.mean(F.relu(1 - real_pred))
        fake_loss = torch.mean(F.relu(1 + fake_pred))
        return real_loss + fake_loss


    def hinge_loss_generator(self, fake_pred):
        """
        Hinge loss for adversarial training of the generator
        """
        return -torch.mean(fake_pred)

    def edge_loss(y_true, y_pred):
        def sobel_filter(x):
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device).unsqueeze(0).unsqueeze(
                0).float()
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device).unsqueeze(0).unsqueeze(
                0).float()
            edges_x = F.conv2d(x, sobel_x, padding=1)
            edges_y = F.conv2d(x, sobel_y, padding=1)
            return torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-6)  # Add small constant to avoid sqrt(0)

        return F.l1_loss(sobel_filter(y_true), sobel_filter(y_pred))


    def compute_gradient_penalty(self, discriminator, damaged_dm, fake_samples, real_samples, max_penalty=1e3):
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

        # Flatten the gradients
        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)  # Add epsilon to avoid NaN

        gradient_penalty = torch.clamp(((gradients_norm - 1) ** 2).mean(), 0, max_penalty)

        return gradient_penalty

