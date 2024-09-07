import time

from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(self, input_images, target_images, seg_maps):
        """
        Args:
            input_images  (list): List of input images
            target_images (list): List of target images
            seg_maps       (list): List of segmentation maps
        """
        self.input_images  = input_images
        self.target_images = target_images
        self.seg_maps      = seg_maps

    def __len__(self):
        return len(self.input_images)

    @staticmethod
    def normalize_tensor(tensor, min_val=0.0, max_val=1.0):
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)

        return normalized_tensor * (max_val - min_val) + min_val

    def __getitem__(self, idx):
        damaged_dm  = self.input_images[idx]
        preserved_dm = self.target_images[idx]
        seg_map      = self.seg_maps[idx]

        # Add normalization here
        # damaged_dm = self.normalize_tensor(damaged_dm)
        # preserved_dm = self.normalize_tensor(preserved_dm)
        # seg_map = self.normalize_tensor(seg_map)

        return damaged_dm, preserved_dm, seg_map

