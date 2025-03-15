import torch
import torch.nn.functional as F
from pytorch_msssim import SSIM


class DHadadLossFunctions:
    """Contains the loss functions used in the DHadad model"""
    def __init__(self, device):
        self.device = device
        self.ssim = SSIM(data_range=1.0, channel=1).to(device)

    @staticmethod
    def l1_loss(pred, target):
        return F.l1_loss(pred, target)

    def ssim_loss(self, pred, target):
        return 1 - self.ssim(pred, target)

    @staticmethod
    def gradient_loss(pred, target):
        def gradient(x):
            h_grad = x[:, :, 1:, :] - x[:, :, :-1, :]
            w_grad = x[:, :, :, 1:] - x[:, :, :, :-1]

            return h_grad, w_grad

        pred_h, pred_w = gradient(pred)
        target_h, target_w = gradient(target)

        gradient_loss = F.l1_loss(pred_h, target_h) + F.l1_loss(pred_w, target_w)

        return gradient_loss

    def depth_range_emphasis_loss(pred, target, min_depth=0.3, max_depth=0.7):
        """Emphasize accurate reconstruction in the typical depth range of inscribed glyphs"""
        # Create mask for regions likely to contain actual glyph data
        mask = ((target >= min_depth) & (target <= max_depth)).float()

        # Apply weighted loss to these regions
        emphasized_loss = F.l1_loss(pred * mask, target * mask, reduction='sum')

        # Normalize by number of pixels in mask
        num_pixels = torch.sum(mask) + 1e-8
        return emphasized_loss / num_pixels

    @staticmethod
    def compute_gradient_penalty(discriminator, damaged_dm, fake_samples, real_samples, max_penalty=1e3):
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

