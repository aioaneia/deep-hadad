
import numpy as np

"""
    Class for dynamically adjusting the weights of the loss function based on the performance of the model.
"""
class DHadadLossWeights:
    """
    The initial weights for the loss function
    
    Test 1:
        SSIM Loss: High importance for perceptual quality                                   - Weight: 0.4
        Geometric Consistency Loss: Critical for maintaining geometric details              - Weight: 0.3
        Depth Consistency Loss (Charbonnier Loss): Essential for smooth depth transitions   - Weight: 0.2
        Edge Loss: Important for edge definition but should not dominate                    - Weight: 0.1
        Sharpness Loss: Use cautiously to avoid introducing noise                           - Weight: 0.05
        Adversarial Loss: (If using GANs)                                                   - Weight: 0.2 (This might need adjustment based on how the GAN is performing)
        L1 Loss: Supporting overall fidelity                                                - Weight: 0.1

    """

    loss_weights_structure = {
        'l1':          0.70, # L1 loss
        'ssim':        0.25, # SSIM loss
        'adversarial': 0.05, # Adversarial loss
        'depth':       0.15, # Depth consistency loss
        'sharp':       0.01, # Sharpness loss
        'geometric':   0.01, # Geometric consistency loss
    }

    loss_weights_details = {
        'l1':          0.60, # L1 loss
        'ssim':        0.25, # SSIM loss
        'adversarial': 0.10, # Adversarial loss
        'depth':       0.15, # Depth consistency loss
        'sharp':       0.5, # Sharpness loss
        'geometric':   0.5, # Geometric consistency loss
    }

    loss_weights_sharp = {
        'l1':          0.30, # L1 loss
        'ssim':        0.20, # SSIM loss
        'adversarial': 0.20, # Adversarial loss
        'depth':       0.15, # Depth consistency loss
        'sharp':       0.10, # Sharpness loss
        'geometric':   0.05, # Geometric consistency loss
    }

    loss_weights_realism = {
        'l1':          0.20, # L1 loss
        'ssim':        0.20, # SSIM loss
        'adversarial': 0.30, # Adversarial loss
        'depth':       0.15, # Depth consistency loss
        'sharp':       0.10, # Sharpness loss
        'geometric':   0.05,  # Geometric consistency loss
    }

    loss_weights_enhanced_realism = {
        'l1':          0.10, # L1 loss
        'ssim':        0.10, # SSIM loss
        'adversarial': 0.40, # Adversarial loss
        'depth':       0.20, # Depth consistency loss
        'sharp':       0.10, # Sharpness loss
        'geometric':   0.10, # Geometric consistency loss
    }

    METRIC_KEY_MAP = {
        'psnr': ['recon'],               # PSNR reflects overall reconstruction quality, influenced by L1 Reconstruction Loss
        'ssim': ['perceptual', 'sharp'], # SSIM captures perceptual quality, influenced by SSIM in Perceptual Loss and Sharpness Loss
        'esi':  ['geometric', 'depth']   # ESI (Edge Similarity Index) impacts Geometric Consistency and Depth Consistency Loss
    }
    
    def __init__(self, weights_type='depth', max_weight=0.5, min_weight=0.05, decay_factor=0.95, max_change=0.05):
        if weights_type == 'depth':
            self.weights = self.loss_weights_structure
        else:
            raise ValueError(f"Invalid weights type: {weights_type}")
        
        self.max_weight        = max_weight
        self.min_weight        = min_weight
        self.max_change        = max_change
        self.decay_factor      = decay_factor
        self.floor             = {key: min_weight for key in self.weights}
        self.cap               = {key: max_weight for key in self.weights}
        self.log               = []
        
    # def update_weights(self, performance_metrics, epoch, total_epochs):
    #     adaptive_decay = self.decay_factor ** (1 + (epoch / total_epochs))
        
    #     # Decrease the weight for the delta loss
    #     magnitude_to_decrease_delta = 0

    #     for metric, info in performance_metrics.items():
    #         keys = self.metric_to_weight_key(metric)

    #         if keys is not None:
    #             for key in keys:
    #                 self.adjust_weight_for_metric(key, info['improved'], info['magnitude'], adaptive_decay)

    #                 magnitude_to_decrease_delta = max(magnitude_to_decrease_delta, info['magnitude'])

    #     self.normalize_weights()

    #     return self.weights
    
    # def adjust_weight_for_metric(self, weight_key, improvement, magnitude, adaptive_decay):
    #     # Base factor for adjustment
    #     factor = adaptive_decay if improvement else (1 / adaptive_decay) * (1 + magnitude)

    #     # Calculate the new weight
    #     new_weight = self.weights[weight_key] * factor

    #     # Ensure the adjusted weight is within bounds and doesn't change too abruptly
    #     change = min(max(new_weight - self.weights[weight_key], -self.max_change), self.max_change)

    #     adjusted_weight = self.weights[weight_key] + change

    #     # Save the old weight 
    #     old_weight = self.weights[weight_key]

    #     # Update the weight
    #     self.weights[weight_key] = max(min(adjusted_weight, self.cap[weight_key]), self.floor[weight_key])

    #     # Log the change
    #     print(f"Weight for {weight_key} changed from {old_weight:.2f} to {self.weights[weight_key]:.2f}")

    # def metric_to_weight_key(self, metric):
    #     """
    #     Maps the metric to the corresponding weight key
    #     """
    #     return self.METRIC_KEY_MAP.get(metric, None)

    def normalize_weights(self):
        """
        Normalizes the weights so that they sum to 1
        """
        total_weight = sum(self.weights.values())

        for key in self.weights:
            self.weights[key] /= total_weight
    
    def get_weights_as_string(self):
        string = ''

        for key, value in self.weights.items():
            # Append a string of the form: _key_value where 
            # key is the first letter of the weight key and value is the weight value
            string += f"_{key[0]}{value:.2f}"
        
        return string

    def manage_epoch_weights(self, epoch):
        """
        Adjusts weights based on the epoch and model performance.
        :param epoch: Current training epoch.
        :param performance_metrics: Dictionary with performance metrics.
        """
        
        if epoch <= 10:
            self.weights = self.loss_weights_structure
        elif 11 < epoch <= 20:
            self.weights = self.loss_weights_details
        elif 21 < epoch <= 30:
            self.weights = self.loss_weights_sharp
        elif 31 < epoch <= 40:
            self.weights = self.loss_weights_realism
        elif 41 < epoch <= 50:
            self.weights = self.loss_weights_enhanced_realism
        
        # Log the changes for analysis
        print(f"Epoch {epoch}: Weights -> {self.weights}")
