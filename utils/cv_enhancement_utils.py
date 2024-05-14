import albumentations as A
import cv2
import numpy as np


def apply_sobel(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    sobel = np.hypot(sobelx, sobely)

    return (sobel / np.max(sobel) * 255).astype(np.uint8)


# Sharpen image
def apply_histogram_equalization(displacement_map):
    """
        Apply histogram equalization to improve the visibility of details
    """

    image = (displacement_map * 255).astype(np.uint8)

    # Create a sharpening kernel
    transform = A.Compose([
        # Contrast Limited Adaptive Histogram Equalization
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1),  # Adaptive histogram equalization
    ])

    # Enhance the depth map
    augmented_image = transform(image=image)['image']

    augmented_image = cv2.normalize(
        augmented_image,
        None,
        alpha=0, beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F)

    return augmented_image
