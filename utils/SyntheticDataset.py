from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(self, input_images, target_images):
        """ Initializes the dataset with the input and target images """
        self.input_images  = input_images
        self.target_images = target_images

    def __len__(self):
        return len(self.input_images)

    @staticmethod
    def normalize_tensor(tensor, min_val=0.0, max_val=1.0):
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)

        return normalized_tensor * (max_val - min_val) + min_val

    def __getitem__(self, idx):
        damaged_dm = self.input_images[idx]
        preserved_dm = self.target_images[idx]

        # damaged_dm = self.normalize_tensor(damaged_dm)
        # preserved_dm = self.normalize_tensor(preserved_dm)

        return damaged_dm, preserved_dm

