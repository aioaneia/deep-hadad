
from torch.utils.data import Dataset

from PIL import Image

class DisplacementMapDataset(Dataset):
    def __init__(self, intact_image_paths, damaged_image_paths, transform=None):
        """
        Args:
            intact_image_paths  (list): List of paths to intact images
            damaged_image_paths (list): List of paths to damaged images
            transform           (callable, optional): Optional transform to be applied on a sample.
        """
        self.intact_image_paths  = intact_image_paths
        self.damaged_image_paths = damaged_image_paths
        self.transform           = transform
        self.counter             = 0 

    def filter_invalid_images(self, image_paths):
        valid_image_paths = []

        for image_path in image_paths:
            try:
                Image.open(image_path)
                valid_image_paths.append(image_path)
            except IOError:
                print(f"Error opening image file: {image_path}")
        
        return valid_image_paths
  
    def __len__(self):
        return len(self.intact_image_paths)

    def __getitem__(self, idx):
        while True:
            try:
                intact_image_path  = self.intact_image_paths[idx]
                damaged_image_path = self.damaged_image_paths[idx]

                intact_image  = Image.open(intact_image_path)
                damaged_image = Image.open(damaged_image_path)

                if self.transform:
                    intact_image  = self.transform(intact_image)
                    damaged_image = self.transform(damaged_image)

                # Increment the counter and log every 100 pairs
                self.counter += 1

                if self.counter % 1000 == 0:
                    print(f"Processed {self.counter} pairs. Current pair: {intact_image_path}, {damaged_image_path}")
                
                return intact_image, damaged_image
            except IOError as e:
                print(f"Error opening image files: {intact_image_path}, {damaged_image_path}. Error: {e}")
                # Increment the index and try the next pair of images
                idx = (idx + 1) % len(self.intact_image_paths)
