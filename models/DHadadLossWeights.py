
class DHadadLossWeights:
    """
    The initial weights for the loss function
    """

    loss_weights_initial = {
        'l1': 1.00,
        'ssim': 0.50,
        'lpips': 0.10,
        'adversarial': 0.05,
        'geometric': 0.10,
    }

    loss_weights_middle = {
        'l1': 0.80,
        'ssim': 0.40,
        'lpips': 0.20,
        'adversarial': 0.10,
        'geometric': 0.10
    }

    loss_weights_final = {
        'l1': 0.60,
        'ssim': 0.30,
        'lpips': 0.30,
        'adversarial': 0.20,
        'geometric': 0.10
    }

    def __init__(self, weights_type='depth', max_weight=0.5, min_weight=0.05, decay_factor=0.95, max_change=0.05):
        if weights_type == 'depth':
            self.weights = self.loss_weights_initial
        else:
            raise ValueError(f"Invalid weights type: {weights_type}")

        self.max_weight = max_weight
        self.min_weight = min_weight
        self.max_change = max_change
        self.decay_factor = decay_factor
        self.floor = {key: min_weight for key in self.weights}
        self.cap = {key: max_weight for key in self.weights}
        self.log = []

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
        """

        if epoch < 20:
            self.weights = self.loss_weights_initial
        elif 11 < 60:
            self.weights = self.loss_weights_middle
        else:
            self.weights = self.loss_weights_final

        # Log the changes for analysis
        print(f"Epoch {epoch}: Weights -> {self.weights}")
