
import numpy as np

"""
    Class for dynamically adjusting the weights of the loss function based on the performance of the model.
"""
class DynamicLossWeights:
    """
    The initial weights for the loss function
    'alpha':      
        # Weight for MSE Loss
        # It ensures the overall structure of the reconstructed image is similar to the intact image.
        # A higher weight for MSE Loss ensures that the overall structure and general content are accurately reconstructed. 
        # This is crucial for readability.
        # MSE could encourage blurred details, which can be detrimental for text recovery and for sharp engravings.
        # Adjustments: If preserving fine details and textures is more important than exact structural matching,
        # consider slightly decreasing alpha (MSE Loss) and increasing beta (L1 Loss).
    'beta':
        # Weight for Reconstruction Loss (L1)
        # It helps in recovering finer details without overly penalizing slight deviations that aren't perceptually significant.
        # Adjustments: If preserving fine details and textures is more important than exact structural matching,
        # consider slightly decreasing alpha (MSE Loss) and increasing beta (L1 Loss).
    'gamma': 
        # Weight for SSIM in Perceptual Loss
        # Captures textural and stylistic features.
        # luminance, and contrast in an image.
        # A higher weight is crucial as it emphasizes on the perceptual similarity, which is key for letter reconstruction.
        # Adjustments: Adjust gamma (SSIM) based on the perceptual quality of the outputs.
        # If the outputs are structurally accurate but lack perceptual quality, consider increasing it.
        # SSIM is important for perceptual quality, ensuring that the textures and details resemble those of genuine inscriptions.
    'delta':
        # Weight for Adversarial Loss
        # TO encourage the generator to create images that are indistinguishable from the intact displacement maps.
        # Encourages realism in the generated maps.
        # Keeping this weight lower ensures that the focus remains on structural and textural accuracy rather than just realism.
        # Adjustments: None
    'epsilon':
        # Weight for Depth Consistency Loss
        # This is a robust loss that combines the benefits of L1 and L2 losses.
        # It can be particularly useful if there's a lot of noise in the damaged maps.
        # Depth data is uncertain in some places
        # This loss can help in smoothing the depth map without losing essential details.
        # Adjustments: If depth and geometric details are crucial,
        # consider slightly increasing epsilon (Depth Consistency Loss) and zeta (Geometric Consistency Loss).
        # This is relevant if your model needs to maintain consistency in the depth aspects of the inscriptions.
    'zeta':
        # Weight for Geometric Consistency Loss
        # Maintains geometric integrity of depth information.
        # This will help in preserving the contours and shapes of letters in the displacement maps.
        # Adjustments: If depth and geometric details are crucial,
        # consider slightly increasing epsilon (Depth Consistency Loss) and zeta (Geometric Consistency Loss).
    eta: 
        Weight for SharpnessLoss
        Encourages sharpness in the generated images.
        This can help in recovering fine details and textures.
        Adjustments: If the images are too blurry or lacking in detail, increase eta (Sharpness Loss).
    'theta':
        Weight for TV Loss
        Encourages spatial smoothness in the generated images.
        This can help in reducing noise and artifacts in the generated images.
        Adjustments: If the images are too smooth or lacking in detail, reduce eta (TV Loss).
    """
    alpha_weights = {
        'alpha':   0.40, # Weight for MSE Loss
        'beta':    0.10, # Weight for Reconstruction Loss (L1)
        'gamma':   0.20, # Weight for SSIM in Perceptual Loss
        'delta':   0.05, # Weight for Adversarial Loss
        'epsilon': 0.10, # Weight for Depth Consistency Loss
        'zeta':    0.10, # Weight for Geometric Consistency Loss
        'eta':     0.05  # Weight for SharpnessLoss
        #'theta':   0.01 # Weight for TV Loss
    }

    gamma_weights = {
        'alpha':   0.10, # Weight for MSE Loss
        'beta':    0.10, # Weight for Reconstruction Loss (L1)
        'gamma':   0.40, # Weight for SSIM in Perceptual Loss
        'delta':   0.10, # Weight for Adversarial Loss
        'epsilon': 0.10, # Weight for Depth Consistency Loss
        'zeta':    0.10, # Weight for Geometric Consistency Loss
        'eta':     0.05  # Weight for SharpnessLoss
        #'theta':   0.01 # Weight for TV Loss
    }

    delta_weights = {
        'alpha':   0.10, # Weight for MSE Loss
        'beta':    0.05, # Weight for Reconstruction Loss (L1)
        'gamma':   0.35, # Weight for SSIM in Perceptual Loss
        'delta':   0.30, # Weight for Adversarial Loss
        'epsilon': 0.10, # Weight for Depth Consistency Loss
        'zeta':    0.05, # Weight for Geometric Consistency Loss
        'eta':     0.05  # Weight for SharpnessLoss
        #'theta':   0.01 # Weight for TV Loss
    }

    delta_weights_stage_2 = {
        'alpha':   0.10, # Weight for MSE Loss
        'beta':    0.05, # Weight for Reconstruction Loss (L1)
        'gamma':   0.35, # Weight for SSIM in Perceptual Loss
        'delta':   0.25, # Weight for Adversarial Loss
        'epsilon': 0.10, # Weight for Depth Consistency Loss
        'zeta':    0.05, # Weight for Geometric Consistency Loss
        'eta':     0.10  # Weight for SharpnessLoss
        #'theta':   0.01 # Weight for TV Loss
    }

    delta_weights_stage_3 = {
        'alpha':   0.10, # Weight for MSE Loss
        'beta':    0.10, # Weight for Reconstruction Loss (L1)
        'gamma':   0.35, # Weight for SSIM in Perceptual Loss
        'delta':   0.15, # Weight for Adversarial Loss
        'epsilon': 0.10, # Weight for Depth Consistency Loss
        'zeta':    0.05, # Weight for Geometric Consistency Loss
        'eta':     0.15  # Weight for SharpnessLoss
        #'theta':   0.01 # Weight for TV Loss
    }

    delta_weights_stage_4 = {
        'alpha':   0.10, # Weight for MSE Loss
        'beta':    0.15, # Weight for Reconstruction Loss (L1)
        'gamma':   0.30, # Weight for SSIM in Perceptual Loss
        'delta':   0.10, # Weight for Adversarial Loss
        'epsilon': 0.15, # Weight for Depth Consistency Loss
        'zeta':    0.05, # Weight for Geometric Consistency Loss
        'eta':     0.15  # Weight for SharpnessLoss
        #'theta':   0.01 # Weight for TV Loss
    }

    zeta_weights = {
        'alpha':   0.15, # Weight for MSE Loss
        'beta':    0.10, # Weight for Reconstruction Loss (L1)
        'gamma':   0.20, # Weight for SSIM in Perceptual Loss
        'delta':   0.25, # Weight for Adversarial Loss
        'epsilon': 0.10, # Weight for Depth Consistency Loss
        'zeta':    0.15, # Weight for Geometric Consistency Loss
        'eta':     0.05  # Weight for SharpnessLoss
        #'theta':   0.01 # Weight for TV Loss
    }

    METRIC_KEY_MAP = {
        'psnr': ['alpha', 'beta'], # PSNR reflects overall reconstruction quality, influenced by MSE (alpha) and L1 (beta)
        'ssim': ['gamma', 'eta'],  # SSIM captures perceptual quality, influenced by SSIM in Perceptual Loss (gamma) and Sharpness (eta)
        'esi':  ['epsilon'] # ESI (Edge Similarity Index) impacts Geometric Consistency (zeta) and Depth Consistency (epsilon)
    }
    
    def __init__(self, weights_type='alpha', max_weight=0.5, min_weight=0.05, decay_factor=0.95, max_change=0.05):
        if weights_type == 'alpha':
            self.weights = self.alpha_weights
        elif weights_type == 'gamma':
            self.weights = self.gamma_weights
        elif weights_type == 'delta':
            self.weights = self.delta_weights
        elif weights_type == 'zeta':
            self.weights = self.zeta_weights
        else:
            raise ValueError(f"Invalid weights type: {weights_type}")
        
        self.max_weight        = max_weight
        self.min_weight        = min_weight
        self.max_change        = max_change
        self.decay_factor      = decay_factor
        self.floor             = {key: min_weight for key in self.weights}
        self.cap               = {key: max_weight for key in self.weights}
        self.log               = []
        
    def update_weights(self, performance_metrics, epoch, total_epochs):
        adaptive_decay = self.decay_factor ** (1 + (epoch / total_epochs))
        
        # Decrease the weight for the delta loss
        magnitude_to_decrease_delta = 0

        for metric, info in performance_metrics.items():
            keys = self.metric_to_weight_key(metric)

            if keys is not None:
                for key in keys:
                    self.adjust_weight_for_metric(key, info['improved'], info['magnitude'], adaptive_decay)

                    magnitude_to_decrease_delta = max(magnitude_to_decrease_delta, info['magnitude'])

        self.normalize_weights()

        return self.weights
    
    def adjust_weight_for_metric(self, weight_key, improvement, magnitude, adaptive_decay):
        # Base factor for adjustment
        factor = adaptive_decay if improvement else (1 / adaptive_decay) * (1 + magnitude)

        # Calculate the new weight
        new_weight = self.weights[weight_key] * factor

        # Ensure the adjusted weight is within bounds and doesn't change too abruptly
        change = min(max(new_weight - self.weights[weight_key], -self.max_change), self.max_change)

        adjusted_weight = self.weights[weight_key] + change

        # Save the old weight 
        old_weight = self.weights[weight_key]

        # Update the weight
        self.weights[weight_key] = max(min(adjusted_weight, self.cap[weight_key]), self.floor[weight_key])

        # Log the change
        print(f"Weight for {weight_key} changed from {old_weight:.2f} to {self.weights[weight_key]:.2f}")

    def metric_to_weight_key(self, metric):
        """
        Maps the metric to the corresponding weight key
        """
        return self.METRIC_KEY_MAP.get(metric, None)

    def normalize_weights(self):
        total_weight = sum(self.weights.values())

        for key in self.weights:
            self.weights[key] = (self.weights[key] / total_weight) * self.max_weight
    
    def get_weights_as_string(self):
        string = ''

        for key, value in self.weights.items():
            # Append a string of the form: _key_value where 
            # key is the first letter of the weight key and value is the weight value
            string += f"_{key[0]}{value:.2f}"
        
        return string
