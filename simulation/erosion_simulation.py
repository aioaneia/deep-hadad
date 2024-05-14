
import cv2
import numpy as np

from scipy.ndimage import gaussian_filter


####################################################################################################
# 2D Image Processing functions for damage simulation
# - Erosion simulation
# - Crack simulation
# - Missing parts simulation
####################################################################################################
def simulate_cv2_erosion(glyph, kernel_size_range=(3, 14), intensity=1.0, iterations=1):
    """
    Simulate erosion of a glyph using OpenCV erode function.

    :param glyph: 2D numpy array with pixel values.
    :param kernel_size_range: The range of the kernel size for erosion in pixels (min, max).
    :param intensity: The intensity of the erosion operation (0.0 to 1.0).
    :param iterations: The number of erosion iterations to apply to the glyph.
    :return: The eroded glyph.
    """

    # Ensure the image is in grayscale
    if len(glyph.shape) > 2:
        glyph = cv2.cvtColor(glyph, cv2.COLOR_BGR2GRAY)

    # Validate and adapt kernel size range based on image size
    max_kernel_size = min(glyph.shape[:2]) // 2

    # Ensure that the kernel size range is within bounds
    kernel_size_range = (max(1, kernel_size_range[0]), min(max_kernel_size, kernel_size_range[1]))

    # Determine kernel size based on intensity
    kernel_size = np.random.randint(*kernel_size_range)
    kernel_size = int(kernel_size * intensity)

    # Define erosion kernel as an elliptical structuring element with the specified kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Apply erosion
    eroded = cv2.erode(glyph, kernel, iterations=iterations)

    return eroded


def top_hat_transform(glyph, kernel_size_range=(3, 14), intensity=1.0, iterations=1):
    """
    Simulate erosion of a glyph using OpenCV morphologyEx function with top hat transform.

    :param glyph: 2D numpy array with pixel values.
    :param kernel_size_range: The range of the kernel size for erosion in pixels (min, max).
    :param intensity: The intensity of the erosion operation (0.0 to 1.0).
    :param iterations: The number of erosion iterations to apply to the glyph.
    :return: The eroded glyph.
    """

    # Ensure the image is in grayscale
    if len(glyph.shape) > 2:
        glyph = cv2.cvtColor(glyph, cv2.COLOR_BGR2GRAY)

    # Validate and adapt kernel size range based on image size
    max_kernel_size = min(glyph.shape[:2]) // 2

    # Ensure that the kernel size range is within bounds
    kernel_size_range = (max(1, kernel_size_range[0]), min(max_kernel_size, kernel_size_range[1]))

    # Determine kernel size based on intensity
    kernel_size = np.random.randint(*kernel_size_range)
    kernel_size = int(kernel_size * intensity)

    # Define kernel as an elliptical structuring element with the specified kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Apply top hat transform
    top_hat = cv2.morphologyEx(glyph, cv2.MORPH_TOPHAT, kernel, iterations=iterations)

    # remove pixel values between 0.7 and 1.0 with the median value
    top_hat[top_hat > 0.7] = np.median(top_hat)

    return top_hat


def measure_glyph_elevation_difference(glyph_depth_map, glyph_mask):
    """
    Measure the elevation difference between the glyph and its surroundings.
    """

    glyph_elevation = np.mean(glyph_depth_map[glyph_mask])
    surroundings_elevation = np.mean(glyph_depth_map[~glyph_mask])
    elevation_difference = glyph_elevation - surroundings_elevation

    return elevation_difference


####################################################################################################
# 3D Point Cloud Processing functions for damage simulation
# - Gaussian erosion simulation
# - Crack simulation
# - Missing parts simulation
####################################################################################################
def simulate_gaussian_erosion_in_point_cloud(point_cloud, erosion_iterations=10, smoothing_sigma=1.0):
    """
    Simulate erosion of a point cloud using Gaussian smoothing.
    """

    eroded_point_cloud = point_cloud.copy()

    for _ in range(erosion_iterations):
        eroded_point_cloud[:, 2] = gaussian_filter(eroded_point_cloud[:, 2], sigma=smoothing_sigma)

    return eroded_point_cloud

