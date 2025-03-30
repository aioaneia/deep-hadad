
class DHadadLossWeights:
    """The initial weights for the loss function"""
    # loss_weights = {
    #     # Early training (establish basic shapes)
    #     'e_0_9':   {'l1': 1.0, 'ssim': 0.3, 'gdl': 2.0, 'adv': 0.1},
    #
    #     # Transition to structural focus
    #     'e_10_19': {'l1': 1.0, 'ssim': 0.4, 'gdl': 3.0, 'adv': 0.2},
    #     'e_20_29': {'l1': 1.0, 'ssim': 0.5, 'gdl': 4.0, 'adv': 0.3},
    #
    #     # Mid-training (balance pixel and structural accuracy)
    #     'e_30_39': {'l1': 0.9, 'ssim': 0.5, 'gdl': 5.0, 'adv': 0.4},
    #     'e_40_59': {'l1': 0.9, 'ssim': 0.5, 'gdl': 5.5, 'adv': 0.5},
    #
    #     # Later training (enhance edges and textures)
    #     'e_60_79': {'l1': 0.8, 'ssim': 0.6, 'gdl': 6.0, 'adv': 0.6},
    #     'e_80_99': {'l1': 0.8, 'ssim': 0.6, 'gdl': 6.5, 'adv': 0.7},
    #
    #     # Final refinement (focus on realism and fine details)
    #     'e_100_119': {'l1': 0.7, 'ssim': 0.7, 'gdl': 7.0, 'adv': 0.8},
    #     'e_120_150': {'l1': 0.7, 'ssim': 0.7, 'gdl': 7.5, 'adv': 1.0}
    # }

    loss_weights = {
        # Phase 1: Geometric Foundation (0-29 epochs)
        'e_0_9':   {'l1': 1.0, 'ssim': 0.8, 'edge': 2.0, 'cont': 1.0, 'adv': 0.1},
        'e_10_19': {'l1': 1.0, 'ssim': 0.7, 'edge': 2.5, 'cont': 1.2, 'adv': 0.1},
        'e_20_29': {'l1': 0.9, 'ssim': 0.6, 'edge': 3.0, 'cont': 1.5, 'adv': 0.2},

        # Phase 2: Detail Refinement (30-89 epochs)
        'e_30_39': {'l1': 0.8, 'ssim': 0.5, 'edge': 3.2, 'cont': 1.8, 'adv': 0.3},
        'e_40_49': {'l1': 0.7, 'ssim': 0.5, 'edge': 3.5, 'cont': 2.0, 'adv': 0.3},
        'e_50_59': {'l1': 0.6, 'ssim': 0.4, 'edge': 3.5, 'cont': 2.2, 'adv': 0.4},
        'e_60_69': {'l1': 0.5, 'ssim': 0.4, 'edge': 3.5, 'cont': 2.5, 'adv': 0.4},
        'e_70_89': {'l1': 0.4, 'ssim': 0.3, 'edge': 3.0, 'cont': 2.8, 'adv': 0.4},

        # Phase 3: Final Polish (90-150 epochs)
        'e_90_109': {'l1': 0.3, 'ssim': 0.2, 'edge': 2.5, 'cont': 3.0, 'adv': 0.5},
        'e_110_150': {'l1': 0.3, 'ssim': 0.1, 'edge': 2.0, 'cont': 3.0, 'adv': 0.5},

        # Phase 4: Advanced Refinement (150+ epochs)
        'e_150_plus': {'l1': 0.3, 'ssim': 0.1, 'edge': 2.0, 'cont': 3.5, 'adv': 0.6}
    }

    def __init__(self, total_epochs=150):
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
        elif epoch < 100:
            self.current_weights = self.loss_weights['e_80_99'].copy()
        elif epoch < 120:
            self.current_weights = self.loss_weights['e_100_119'].copy()
        elif epoch < 150:
            self.current_weights = self.loss_weights['e_120_150'].copy()
        else:
            self.current_weights = self.loss_weights['e_150_plus'].copy()

        print(f"Epoch {epoch}: Weights -> {self.current_weights}")
