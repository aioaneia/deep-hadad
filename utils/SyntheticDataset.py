import time

from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(self, input_images, target_images):
        """
        Args:
            input_images  (list): List of input images
            target_images (list): List of target images
        """

        self.counter = 0
        self.count_skipped = 0
        self.start_time = time.time()

        self.input_images = input_images
        self.target_images = target_images

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = self.input_images[idx]
        target_image = self.target_images[idx]

        # Increment the counter and log every 100 pairs
        self.counter += 1

        # Log every 500 pairs
        if self.counter % 100 == 0:
            print(f"Number of image pairs processed: {self.counter}")

            # Print the file paths of the input and target images
            # print(f"Input image: {input_image}")
            # print(f"Target image: {target_image}")

            # Calculate the time per image pair and log it with two decimal places
            time_per_image_pair = (time.time() - self.start_time) / self.counter
            print(f"Time per image pair: {time_per_image_pair:.2f} seconds")

        return input_image, target_image

