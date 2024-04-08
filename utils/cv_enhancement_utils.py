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
def sharp_image(image):
    """
    Sharpen the image using the unsharp mask method.
    """

    # Create a sharpening kernel
    transform = A.Compose([
        # Contrast Limited Adaptive Histogram Equalization
        A.CLAHE(clip_limit=(2.0, 3.0), p=1),

        # Gamma Contrast
        A.RandomGamma(gamma_limit=(90, 110), p=1),

        # Sharpen the image
        A.Sharpen(alpha=(0.9, 1), lightness=(1.0, 1.0), p=1),

        # Emboss effect to enhance texture
        A.Emboss(alpha=(0.4, 0.6), strength=(0.9, 1), p=1),
    ])

    # Enhance the depth map
    augmented_image = transform(image=image)['image']

    return augmented_image