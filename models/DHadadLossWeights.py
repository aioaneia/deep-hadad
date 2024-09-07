
class DHadadLossWeights:
    """
    The initial weights for the loss function
    """
    loss_weights = {
        'e_0_9':     {'l1': 100.0, 'ssim': 10.0, 'ms_ssim': 5.0,  'gdl': 10.0, 'tv': 0.1, 'freq': 5.0,  'adv': 0.4},
        'e_10_19':   {'l1': 95.0,  'ssim': 12.0, 'ms_ssim': 6.0,  'gdl': 10.0, 'tv': 0.2, 'freq': 6.0,  'adv': 0.4},
        'e_20_29':   {'l1': 90.0,  'ssim': 14.0, 'ms_ssim': 7.0,  'gdl': 12.0, 'tv': 0.3, 'freq': 7.0,  'adv': 0.6},
        'e_30_39':   {'l1': 85.0,  'ssim': 16.0, 'ms_ssim': 8.0,  'gdl': 14.0, 'tv': 0.4, 'freq': 8.5,  'adv': 0.8},
        'e_40_59':   {'l1': 80.0,  'ssim': 18.0, 'ms_ssim': 9.0,  'gdl': 16.0, 'tv': 0.5, 'freq': 9.0,  'adv': 1.0},
        'e_60_79':   {'l1': 75.0,  'ssim': 20.0, 'ms_ssim': 10.0, 'gdl': 18.0, 'tv': 0.6, 'freq': 10.5, 'adv': 1.4},
        'e_80_99':   {'l1': 70.0,  'ssim': 22.0, 'ms_ssim': 11.0, 'gdl': 20.0, 'tv': 0.7, 'freq': 11.0, 'adv': 1.8},
        'e_100_119': {'l1': 65.0,  'ssim': 24.0, 'ms_ssim': 12.0, 'gdl': 22.0, 'tv': 0.8, 'freq': 12.0, 'adv': 2.2},
        'e_120_139': {'l1': 60.0,  'ssim': 26.0, 'ms_ssim': 13.0, 'gdl': 24.0, 'tv': 0.9, 'freq': 13.5, 'adv': 2.6},
    }

    def __init__(self, total_epochs=100):
        self.total_epochs    = total_epochs
        self.current_weights = self.loss_weights['e_0_9'].copy()


    def get_weights_as_string(self):
        string = ''

        for key, value in self.current_weights.items():
            # Append a string of the form: _key_value where 
            # key is the first letter of the weight key and value is the weight value
            string += f"_{key[0]}{value:.2f}"

        return string

    def manage_epoch_weights(self, epoch):
        if epoch < 10:
            self.current_weights = self.loss_weights['e_0_9'].copy()
        elif epoch < 20:
            self.current_weights = self.loss_weights['e_10_19'].copy()
        elif epoch < 30:
            self.current_weights = self.loss_weights['e_20_29'].copy()
        elif epoch < 40:
            self.current_weights = self.loss_weights['e_30_39'].copy()
        elif epoch < 60:
            self.current_weights = self.loss_weights['e_40_59'].copy()
        elif epoch < 80:
            self.current_weights = self.loss_weights['e_60_79'].copy()
        else:
            self.current_weights = self.loss_weights['e_80_99'].copy()

        print(f"Epoch {epoch}: Weights -> {self.current_weights}")
