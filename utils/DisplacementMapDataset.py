import time

from torch.utils.data import Dataset

import utils.cv_file_utils as file_utils


class DisplacementMapDataset(Dataset):
    def __init__(self, input_image_paths, target_image_paths, transform=None, preload_images=True):
        """
        Args:
            input_image_paths  (list): List of paths for the input images
            targer_image_paths (list): List of paths for the target images
            transform          (callable, optional): Optional transform to be applied on a sample.
        """
        self.input_image_paths = input_image_paths
        self.target_image_paths = target_image_paths
        self.transform = transform
        self.preload_images = preload_images

        self.counter = 0
        self.count_skipped = 0
        self.start_time = time.time()

        if preload_images:
            self.input_images = [file_utils.load_displacement_map_as_tensor(path) for path in input_image_paths]
            self.target_images = [file_utils.load_displacement_map_as_tensor(path) for path in target_image_paths]
        else:
            self.input_images = None
            self.target_images = None

    def __len__(self):
        return len(self.input_image_paths)

    def __getitem__(self, idx):
        while True:
            try:
                input_image_path = self.input_image_paths[idx]
                target_image_path = self.target_image_paths[idx]

                if self.preload_images:
                    input_image = self.input_images[idx]
                    target_image = self.target_images[idx]
                else:
                    input_image = file_utils.load_displacement_map_as_tensor(input_image_path)
                    target_image = file_utils.load_displacement_map_as_tensor(target_image_path)

                    # Print the shapes of the tensors before appending to the dataset
                    # print("Input Displacement Map Tensor Shape:", input_image.shape)
                    # print("Target Displacement Map Tensor Shape:", target_image.shape)

                # Check if the images were loaded successfully
                if input_image is None or target_image is None:
                    print(f"Error opening image files:       {input_image_path}, {target_image_path}")
                    print(f"Skipping image pair:             {input_image_path}, {target_image_path}")
                    print(f"Number of image pairs processed: {self.counter}")
                    print(f"Number of image pairs remaining: {len(self.input_image_paths) - self.counter}")
                    print(f"Number of image pairs skipped:   {self.count_skipped}")

                    # Increment the index and try the next pair of images
                    idx = (idx + 1) % len(self.input_image_paths)

                    # Increment the number of skipped images
                    self.count_skipped += 1

                    continue

                # if self.transform:
                #     input_image  = self.transform(input_image)
                #     target_image = self.transform(target_image)

                # Increment the counter and log every 100 pairs
                self.counter += 1

                # Log every 500 pairs
                if self.counter % 50 == 0:
                    print(f"Processed image pair: {input_image_path}, {target_image_path}")
                    print(f"Number of image pairs processed: {self.counter}")
                    print(f"Number of image pairs skipped:   {self.count_skipped}")

                    # Calculate the time per image pair and log it with two decimal places
                    time_per_image_pair = (time.time() - self.start_time) / self.counter
                    print(f"Time per image pair:             {time_per_image_pair:.2f} seconds")

                return input_image, target_image
            except IOError as e:
                print(f"Error opening image files: {input_image_path}, {target_image_path}. Error: {e}")
                # Increment the index and try the next pair of images
                idx = (idx + 1) % len(self.input_image_paths)

                # Increment the number of skipped images
                self.count_skipped += 1
