
from torch.utils.data import Dataset

from PIL import Image

import cv2

class DisplacementMapDataset(Dataset):
    def __init__(self, input_image_paths, target_image_paths, transform=None):
        """
        Args:
            input_image_paths  (list): List of paths for the input images
            targer_image_paths (list): List of paths for the target images
            transform          (callable, optional): Optional transform to be applied on a sample.
        """
        self.input_image_paths  = input_image_paths
        self.target_image_paths = target_image_paths
        self.transform          = transform
        self.counter            = 0 

    def filter_invalid_images(self, image_paths):
        """
        Filters out invalid images from the list of image paths
        :param image_paths: The list of image paths
        :return: The list of valid image paths
        """
        valid_image_paths = []

        for image_path in image_paths:
            try:
                Image.open(image_path)
                valid_image_paths.append(image_path)
            except IOError:
                print(f"Error opening image file: {image_path}")
        
        return valid_image_paths
  
    def __len__(self):
        return len(self.input_image_paths)

    def __getitem__(self, idx):
        while True:
            try:
                input_image_path   = self.input_image_paths[idx]
                target_image_path = self.target_image_paths[idx]

                # Load the input and target images as grayscale images represented as NumPy arrays
                input_image  = cv2.imread(input_image_path,  cv2.IMREAD_GRAYSCALE)
                target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)

                # Convert the NumPy array to a PIL Image 
                input_image  = Image.fromarray(input_image)
                target_image = Image.fromarray(target_image)

                if self.transform:
                    input_image   = self.transform(input_image)
                    target_image = self.transform(target_image)

                # Increment the counter and log every 100 pairs
                self.counter += 1

                # Log every 3000 pairs of images processed the current pair of images
                if self.counter % 3000 == 0:
                    print(f"Processed {self.counter} pairs. Current pair: {input_image_path}, {target_image_path}")
                
                return input_image, target_image
            except IOError as e:
                print(f"Error opening image files: {input_image_path}, {target_image_path}. Error: {e}")
                # Increment the index and try the next pair of images
                idx = (idx + 1) % len(self.input_image_paths)
