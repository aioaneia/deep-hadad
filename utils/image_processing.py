import cv2
import numpy as np
from scipy.stats import ks_2samp


####################################################################################################
# Add blur
# # Randomly choose between 3x3, 5x5, 7x7 kernel sizes
####################################################################################################
def add_blur(image, kernel_size=3, sigma=0, preserve_edges=True):
    if preserve_edges:
        # Use bilateral filtering to preserve edges. 
        # Adjust sigmaColor and sigmaSpace.
        blurred_map = cv2.bilateralFilter(image, d=kernel_size, sigmaColor=75, sigmaSpace=75)
    else:
        # Apply standard Gaussian blur, 
        # allowing sigma to be calculated from the kernel size if sigma=0
        blurred_map = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)

    return blurred_map


def add_adaptive_blur(image, kernel_size=3, low_threshold=50, high_threshold=150):
    """
    Applies adaptive Gaussian blur to an inscription map to enhance eroded letters
    while preserving essential details.

    Parameters:
    - image: The input map of the ancient inscription.
    - max_kernel_size: The maximum size of the Gaussian kernel for the blur.

    Returns:
    - The blurred inscription map.
    """

    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    # Create a copy of the map to work on
    blurred_image = gray_image.copy()

    # Detect edges to preserve them; the threshold values may be adjusted for displacement maps
    edges = cv2.Canny(gray_image, threshold1=low_threshold, threshold2=high_threshold)

    # Invert edges to create a mask for areas to blur
    mask = cv2.bitwise_not(edges)

    # Use adaptiveThreshold to identify regions with uniform intensity
    adaptive_mask = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, kernel_size, 2
    )

    # Combine masks to exclude edges and uniform regions from blurring
    combined_mask = cv2.bitwise_or(adaptive_mask, mask)

    # Apply a Gaussian blur on the entire image
    full_blur = cv2.GaussianBlur(blurred_image, (kernel_size, kernel_size), 0)

    # Where combined_mask is not set, replace blurred_image with full_blur
    blurred_image = np.where(combined_mask[..., None], blurred_image, full_blur)

    return blurred_image.squeeze().copy()


####################################################################################################
# Add noise
####################################################################################################
def add_noise(image, intensity=0.5):
    # Randomly choose between Gaussian and Salt & Pepper noise
    noise_type = np.random.choice(['gaussian', 's&p'])

    if noise_type == 'gaussian':
        mean = 0
        sigma = intensity  # Adjust intensity based on the map's range
        gauss = np.random.normal(mean, sigma, image.shape)
        noisy = image + gauss
    else:
        # Salt & Pepper
        s_vs_p = np.random.uniform(0.3, 0.7)
        amount = intensity / 10  # Adjust to be lower for displacement maps
        noisy = np.copy(image)

        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p).astype(int)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
        noisy[tuple(coords)] = image.max()

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p)).astype(int)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
        noisy[tuple(coords)] = image.min()

    # Clip to ensure the noisy image is within the valid range
    noisy = np.clip(noisy, image.min(), image.max()).astype(image.dtype)

    return noisy


####################################################################################################
# Add dilate image
# 
####################################################################################################
def dilate_image(image, kernel_size_range=(1, 5), intensity=1.0, inscription_mask=None, iterations=1):
    # Randomly choose kernel size
    kernel_size = np.random.randint(*kernel_size_range)

    # Define dilation 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # If an inscription mask is provided, apply dilation selectively
    if inscription_mask is not None:
        masked_image = cv2.bitwise_and(image, image, mask=inscription_mask)

        dilated = cv2.dilate(masked_image, kernel, iterations=iterations)

        # Combine the dilated inscriptions with the original image
        dilated = cv2.max(image, dilated)
    else:
        dilated = cv2.dilate(image, kernel, iterations=iterations)

    return dilated


####################################################################################################
# The function takes an image as input and returns a skewed version of the image.
# Referece: https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
####################################################################################################
def skew_image(image, x_skew_range=(0.7, 1.3), y_skew_range=(0.7, 1.5)):
    # Skew image
    rows, cols = image.shape
    # Randomly choose skew factors
    x_skew = np.random.uniform(*x_skew_range)
    y_skew = np.random.uniform(*y_skew_range)

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[50 * x_skew, 50 * y_skew], [200 * x_skew, 50], [50, 200 * y_skew]])

    M = cv2.getAffineTransform(pts1, pts2)
    skewed = cv2.warpAffine(image, M, (cols, rows))
    return skewed


####################################################################################################
# stretch image
####################################################################################################
def stretch_image(image, x_factor_range=(0.8, 1.3), y_factor_range=None):
    # Generate factors
    x_factor = np.random.uniform(*x_factor_range)
    y_factor = np.random.uniform(*y_factor_range) if y_factor_range else x_factor

    # Stretch image
    return cv2.resize(image, None, fx=x_factor, fy=y_factor, interpolation=cv2.INTER_AREA)


####################################################################################################
# adjust brightness
####################################################################################################
def adjust_brightness(image, brightness_factor_range=(0.7, 1.3)):
    # Randomly choose brightness factor
    brightness_factor = np.random.uniform(*brightness_factor_range)

    # Adjust brightness
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v * brightness_factor, 0, 1).astype(np.float32)
    return cv2.merge((h, s, v))


####################################################################################################
# Simulate text fading
####################################################################################################
def simulate_text_fading(image, num_areas_range=(1, 5), area_size_range=(10, 40), fading_intensity_range=(0.3, 0.9)):
    # Convert image to float32 for processing
    faded_image = image.astype(np.float32)

    num_areas = np.random.randint(*num_areas_range)

    for _ in range(num_areas):
        w, h = np.random.randint(*area_size_range, size=2)
        x, y = np.random.randint(0, max(1, image.shape[1] - w)), np.random.randint(0, max(1, image.shape[0] - h))
        fading_intensity = np.random.uniform(*fading_intensity_range)

        # Creating a radial gradient
        y_indices, x_indices = np.ogrid[:h, :w]
        center_x, center_y = w / 2, h / 2
        distance = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        gradient = 1 - (distance / max_dist)

        # Apply fading with gradient
        faded_image[y:y + h, x:x + w] *= fading_intensity + (1 - fading_intensity) * gradient

    # Convert back to original data type
    faded_image = np.clip(faded_image, 0, 255).astype(image.dtype)

    return faded_image


####################################################################################################
# Simulate bumps and scratches
####################################################################################################
def simulate_bumps_and_scratches(image, intensity=0.5, scratches=True, bumps=False):
    # Create a copy of the image to work on
    simulated_image = image.copy()

    # Calculate the number of features based on intensity
    num_bumps = int(6 * intensity)
    num_scratches = int(6 * intensity)

    if bumps == True:
        # Adding bumps
        for _ in range(num_bumps):
            x, y = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])
            radius = np.random.randint(1, 6)
            bump_mask = np.zeros_like(image)

            cv2.circle(bump_mask, (x, y), radius, (0), -1)

            bump_blurred = cv2.GaussianBlur(bump_mask, (0, 0), radius / 2)
            simulated_image = cv2.addWeighted(simulated_image, 1, bump_blurred, intensity, 0)

    if scratches == True:
        # Adding scratches
        for _ in range(num_scratches):
            x_start, y_start = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])
            x_end, y_end = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])
            thickness = np.random.randint(1, 3)

            cv2.line(simulated_image, (x_start, y_start), (x_end, y_end), (0), thickness)

    return simulated_image


####################################################################################################
# Simulate discoloration and texture
####################################################################################################
def simulate_discoloration_and_texture(image, discoloration_intensity=0.5, texture_intensity=0.9, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    # Discoloration
    if discoloration_intensity > 0:
        discoloration_mask = np.random.uniform(1 - discoloration_intensity, 1, size=image.shape)
        image = (image * discoloration_mask).astype(np.uint8)

    # Texture Alteration
    if texture_intensity > 0:
        num_regions = np.random.randint(1, 4)
        for _ in range(num_regions):
            x, y = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])
            w, h = np.random.randint(10, 30), np.random.randint(10, 30)
            blur_size = int(7 * texture_intensity)
            blur_size = blur_size + 1 if blur_size % 2 == 0 else blur_size  # ensuring odd size for Gaussian kernel
            smoothed_region = cv2.GaussianBlur(image[y:y + h, x:x + w], (blur_size, blur_size), 0)
            image[y:y + h, x:x + w] = smoothed_region

    return image


####################################################################################################
# Validate Intensity Distribution
####################################################################################################
def validate_intensity_distribution(real_images, synthetic_images):
    synthetic_intensities = [img.mean() for img in synthetic_images]
    real_intensities = [img.mean() for img in real_images]

    # Compare distributions, e.g., using Kolmogorov-Smirnov test
    ks_statistic, p_value = ks_2samp(synthetic_intensities, real_intensities)

    if p_value < 0.05:
        print("Distributions differ significantly.")
        return False
    else:
        print("No significant difference in distributions.")
        return True

