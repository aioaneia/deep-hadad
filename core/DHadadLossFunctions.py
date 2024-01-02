
import torch
import torch.nn            as nn
import torch.nn.functional as F

from loss.WeightedSSIMLoss         import WeightedSSIMLoss
from loss.DepthConsistencyLoss     import DepthConsistencyLoss
from loss.GeometricConsistencyLoss import GeometricConsistencyLoss
from loss.SharpnessLoss            import SharpnessLoss

class DHadadLossFunctions:
    """
    Contains the loss functions used in the DHadad model
    """

    @staticmethod
    def mean_sq_error_loss(input, target):
        """
        Calculates the mean squared error loss for a batch of images
        
        Pros:
            It ensures the overall structure of the reconstructed image
            is similar to the intact image.
        Cons:
            MSE could encourage blurred details, which can be detrimental 
            for text recovery and for sharp engravings.

        :param input: The input images
        :param target: The target images
        :return: The mean squared error loss for the batch
        """
        return nn.MSELoss()(input, target)
    
    @staticmethod
    def l1_loss(input, target):
        """
        Calculates the L1 loss for a batch of images
        
        Pros:
            It helps in recovering finer details without overly penalizing slight deviations that 
            aren't perceptually significant.
        Cons:
            It can be detrimental for sharp engravings.

        :param input: The input images
        :param target: The target images
        :return: The L1 loss for the batch
        """
        return nn.L1Loss()(input, target)

    @staticmethod
    def ssim_loss(input, target):
        """
        Calculates the SSIM loss for a batch of images
        
        Pros:
            It captures textural and stylistic features.
            it captures the perceptual similarity between the images.
            luminance, and contrast in an image.
            It's a better alternative to MSE and L1 losses.
        Cons:
            It can be detrimental for sharp engravings.

        :param input: The input images
        :param target: The target images
        :return: The SSIM loss for the batch
        """
        weighted_ssim_loss = WeightedSSIMLoss(data_range = 1, size_average = True, channel = 1, weight = 1.0)
        
        return weighted_ssim_loss(input, target)

    @staticmethod
    def dce_with_logits_loss(input, target):
        """
        Calculates the binary cross entropy with logits loss for a batch of images
        TO encourage the generator to create images that are indistinguishable
        from the intact displacement maps.
        Encourages realism in the generated maps.

        Pros:
            It's a good loss function for binary classification problems.
        Cons:
            It can be detrimental for sharp engravings.

        :param input: The input images
        :param target: The target images
        :return: The binary cross entropy with logits loss for the batch
        """
        return nn.BCEWithLogitsLoss()(input, target)
    
    @staticmethod
    def depth_consistency_loss(input, target):
        """
        Calculates the depth consistency loss for a batch of images
        
        This is a robust loss that combines the benefits of L1 and L2 losses.
        It can be particularly useful if there's a lot of noise in the damaged maps.
        Depth data is uncertain in some places
        This loss can help in smoothing the depth map without losing essential details.

        Pros:
            It's a good loss function for binary classification problems.
        Cons:
            It can be detrimental for sharp engravings.

        :param input: The input images
        :param target: The target images
        :return: The depth consistency loss for the batch
        """
        return DepthConsistencyLoss()(input, target)

    @staticmethod
    def geometric_consistency_loss(input, target):
        """
        Calculates the geometric consistency loss for a batch of images
        
        Maintains geometric integrity of depth information.
        This will help in preserving the contours and shapes of letters in the displacement maps.

        Pros:
            It's a good loss function for binary classification problems.
        Cons:
            It can be detrimental for sharp engravings.

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
        This can help in reducing noise and artifacts in the generated images.

        Pros:
            It's a good loss function for binary classification problems.
        Cons:
            It can be detrimental for sharp engravings.

        :param input: The input images
        :param target: The target images
        :return: The sharpness loss for the batch
        """
        return SharpnessLoss()(input, target)

    @staticmethod
    def tv_loss(img):
        """
        Calculates the total variation loss for a batch of images
        
        Encourages spatial smoothness in the generated images.
        This can help in reducing noise and artifacts in the generated images.

        Pros:
            It's a good loss function for binary classification problems.
        Cons:
            It can be detrimental for sharp engravings.

        :param input: The input images
        :param target: The target images
        :return: The total variation loss for the batch
        """

        batch_size, _, height, width = img.size()
        
        tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()

        return (tv_h + tv_w) / (batch_size * height * width)


    @staticmethod
    def adversarial_loss(discriminator_preds, is_real=True):
        """
        Calculates adversarial loss for the discriminator or generator output.
        Encourages the generator to create images that are indistinguishable
        from the intact displacement maps.

        Pros:
            It's a good loss function for binary classification problems.
        Cons:
            It can be detrimental for sharp engravings.
        
        :param discriminator_preds: Predictions from the discriminator
        :param is_real: Flag indicating whether the target is real or fake
        :return: Adversarial loss value
        """

        # Dynamically determine the device from the input tensors
        device = discriminator_preds.device

        # Create labels based on whether is_real is True or False
        labels = torch.ones_like(discriminator_preds, device=device) if is_real \
            else torch.zeros_like(discriminator_preds, device=device)

        # Calculate Binary Cross Entropy with Logits Loss
        return nn.BCEWithLogitsLoss()(discriminator_preds, labels)


    def __init__(self):
        pass

    def calculate_gen_loss(self, fake_imgs, real_imgs, discriminator_preds, loss_weights):
        """
        Calculates the combined generator loss
        """
        # Mean Squared Error Loss
        #mse_loss = self.mean_sq_error_loss(fake_imgs, real_imgs)

        # Reconstruction Loss (L1)
        recon_loss = self.l1_loss(fake_imgs, real_imgs)

        # Perceptual Loss
        ssim_loss = self.ssim_loss(fake_imgs, real_imgs)

        # Adversarial Loss for the generator
        adv_loss = self.adversarial_loss(discriminator_preds, is_real=False)

        # Charbonnier Loss (robust L1/L2 loss)
        depth_loss = self.depth_consistency_loss(fake_imgs, real_imgs)

        # predicted_map and target_map are your generated and ground truth displacement maps, respectively
        geom_loss = self.geometric_consistency_loss(fake_imgs, real_imgs)

        sharp_loss = self.sharpness_loss(fake_imgs, real_imgs)

        # Check for NaNs in losses
        if any(torch.isnan(loss).any() for loss in [recon_loss, ssim_loss, adv_loss, depth_loss, geom_loss, sharp_loss]):
            print("NaNs detected in generator losses")

        # Combine the losses
        combined_loss = ( # torch.mean
            loss_weights['recon']       * recon_loss + \
            loss_weights['perceptual']  * ssim_loss + \
            loss_weights['adversarial'] * adv_loss + \
            loss_weights['depth']       * depth_loss + \
            loss_weights['geometric']   * geom_loss + \
            loss_weights['sharp']       * sharp_loss
        )

        return combined_loss
    

    ########################################################################################
    # Define Evaluation Functions
    # Gradient Penalty
    # The gradient penalty is typically used in the context of Wasserstein GANs
    # with Gradient Penalty (WGAN-GP).
    # It enforces the Lipschitz constraint by penalizing the gradient norm
    # of the discriminator's output with respect to its input.
    ########################################################################################
    def compute_gradient_penalty(self, discriminator, fake_samples, real_samples):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=real_samples.device)

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        
        # Logging shapes and checking for NaNs
        #print("Interpolates Shape: ", interpolates.shape)

        if torch.isnan(interpolates).any():
            print("NaNs in interpolates")
            #return torch.tensor(0.0).to(real_samples.device)  # Early return with a default value

        d_interpolates = discriminator(interpolates)

        if d_interpolates.grad_fn is None:
            print("d_interpolates does not have a valid grad_fn")
            #return torch.tensor(0.0).to(real_samples.device)

        # Checking for NaNs after discriminator
        if torch.isnan(d_interpolates).any():
            print("NaNs in discriminator output")
            #return torch.tensor(0.0).to(real_samples.device)  # Early return with a default value
        
        #fake = torch.ones(d_interpolates.size(), requires_grad=False, device=real_samples.device)
        grad_outputs = torch.ones_like(d_interpolates)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs      = d_interpolates,
            inputs       = interpolates,
            grad_outputs = grad_outputs,
            create_graph = True,
            retain_graph = True,
            only_inputs  = True
            #allow_unused = True # This allows for the case where some inputs might not affect outputs
        )[0]

        # Handling the case where gradients might be None
        if gradients is None:
            print("No gradients found for interpolates")
            #return torch.tensor(0.0).to(real_samples.device)
        
        # Checking for NaNs in gradients
        if torch.isnan(gradients).any():
            print("NaNs in gradients")
            #return torch.tensor(0.0).to(real_samples.device)  # Early return with a default value

        gradients        = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        # Checking for NaNs in gradient penalty
        if torch.isnan(gradient_penalty):
            print("NaNs in gradient penalty")
            #return torch.tensor(0.0).to(real_samples.device)  # Early return with a default value

        return gradient_penalty

    
    def calculate_dis_loss(self, discriminator, fake_imgs, real_imgs, lambda_gp):
        """
        Calculates the combined discriminator loss
        """

        # Classify fake image
        # The enhanced_dm is the fake image here
        output_fake = discriminator(fake_imgs)

        # Classify the real image
        # The synthetic_dm is the real image here
        output_real = discriminator(real_imgs) 

        # Adversarial Loss for the discriminator
        real_loss = self.adversarial_loss(output_real, is_real=True)

        # Adversarial Loss for the generator
        fake_loss = self.adversarial_loss(output_fake, is_real=False)

        # Gradient Penalty
        gradient_penalty = self.compute_gradient_penalty(discriminator, fake_imgs, real_imgs) * lambda_gp

        # Check for NaNs in losses
        if any(torch.isnan(loss).any() for loss in [real_loss, fake_loss]):
            print("NaNs detected in discriminator losses")

        # Compute the total discriminator loss
        combined_loss = (real_loss + fake_loss + lambda_gp * gradient_penalty) / 3

        return combined_loss
