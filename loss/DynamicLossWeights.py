
"""
    Class for dynamically adjusting the weights of the loss function based on the performance of the model.
"""
class DynamicLossWeights:
    """
    The mapping of the performance metrics to the corresponding weight keys
    'alpha':   Weight for MSE Loss
    'beta':    Weight for Reconstruction Loss (L1)
    'gamma':   Weight for SSIM in Perceptual Loss
    'delta':   Weight for Adversarial Loss
    'epsilon': Weight for Depth Consistency Loss
    'zeta':    Weight for Geometric Consistency Loss
    'eta':     Weight for TV Loss
    """
    METRIC_KEY_MAP = {
        'esi':  ['zeta', 'epsilon'],
        'psnr': ['alpha', 'beta'], 
        'ssim': ['gamma'] 
    }
    
    def __init__(self, initial_weights, max_weight=1.0, min_weight=0.01, base_decay_factor=0.95):
        self.weights           = initial_weights
        self.max_weight        = max_weight
        self.min_weight        = min_weight
        self.base_decay_factor = base_decay_factor
        self.floor             = {key: min_weight for key in initial_weights}
        self.cap               = {key: max_weight for key in initial_weights}
        self.log               = []

    def update_weights(self, performance_metrics, epoch, total_epochs):
        adaptive_decay = self.base_decay_factor ** (1 + (epoch / total_epochs))
        
        for metric, info in performance_metrics.items():
            keys = self.metric_to_weight_key(metric)

            if keys is not None:
                for key in keys:
                    self.adjust_weight_for_metric(key, info['improved'], info['magnitude'], adaptive_decay)

        return self.weights

    def adjust_weight_for_metric(self, weight_key, improvement, magnitude, adaptive_decay):
        factor = adaptive_decay if improvement else (1 / adaptive_decay) * (1 + magnitude)

        new_weight = self.weights[weight_key] * factor

        self.weights[weight_key] = max(min(new_weight, self.cap[weight_key]), self.floor[weight_key])

        # Logging for analysis
        self.log.append((weight_key, self.weights[weight_key]))

    
    def metric_to_weight_key(self, metric):
        """
        Maps the metric to the corresponding weight key

        :param metric: The metric to map
        :return: The weight keys
        """
        return self.METRIC_KEY_MAP.get(metric, None)

    def normalize_weights(self):
        total_weight = sum(self.weights.values())

        for key in self.weights:
            self.weights[key] = (self.weights[key] / total_weight) * self.max_weight
    
    def get_weights_as_string(self):
        string = ''

        for key, value in self.weights.items():
            string += '_' + key + '_' + str(round(value, 3))
        
        return string
