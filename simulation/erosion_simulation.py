import cv2
import numpy as np


class DepthAwareErosion:
    def __init__(self, depth_map):
        self.depth_map = depth_map

    def apply(self):
        # Deeper regions erode faster
        erosion_strength = np.clip(self.depth_map * 2.5, 1, 10)

        kernel_size = int(5 + erosion_strength.mean())

        return cv2.erode(self.depth_map, kernel=np.ones((kernel_size,kernel_size)))


def simulate_cv2_erosion(glyph, kernel_size_range=(3, 14), intensity=1.0, iterations=1):
    """
    Simulate erosion of a glyph using OpenCV erode function.

    :param glyph: 2D numpy array with pixel values.
    :param kernel_size_range: The range of the kernel size for erosion in pixels (min, max).
    :param intensity: The intensity of the erosion operation (0.0 to 1.0).
    :param iterations: The number of erosion iterations to apply to the glyph.
    :return: The eroded glyph.
    """
    # Validate and adapt kernel size range based on image size
    max_kernel_size = min(glyph.shape[:2]) // 2

    # Ensure that the kernel size range is within bounds
    kernel_size_range = (max(1, kernel_size_range[0]), min(max_kernel_size, kernel_size_range[1]))

    # Determine kernel size based on intensity
    kernel_size = np.random.randint(*kernel_size_range)
    kernel_size = int(kernel_size * intensity)

    # Ensure kernel size is a positive odd integer
    kernel_size = max(1, kernel_size | 1)  # This ensures kernel_size is odd

    # Create a non-uniform erosion mask
    erosion_mask = np.random.rand(*glyph.shape) * intensity
    erosion_mask = cv2.GaussianBlur(erosion_mask, (kernel_size, kernel_size), 0)

    # Apply non-uniform erosion
    eroded = np.zeros_like(glyph, dtype=np.float32)

    # Define erosion kernel as an elliptical structuring element with the specified kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Apply erosion
    eroded = cv2.erode(glyph, kernel, iterations=iterations)

    return eroded
