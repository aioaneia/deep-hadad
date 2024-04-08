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


def simulate_crack(glyph_depth_map, crack_depth_map):
    """
    Simulate a crack in the glyph depth map by using a crack depth map.

    :param glyph_depth_map: 2D numpy array with depth values of the glyph.
    :param crack_depth_map: 2D numpy array with depth values of the crack (must be same scale).
    :return: Depth map with a simulated crack.
    """
    # Resize the crack depth map to match the glyph depth map's size
    resized_crack_depth_map = cv2.resize(
        crack_depth_map,
        (glyph_depth_map.shape[1], glyph_depth_map.shape[0]),
        interpolation=cv2.INTER_NEAREST)

    # Invert the crack depth map so that cracks are 'deep'
    inverted_crack_depth_map = np.max(resized_crack_depth_map) - resized_crack_depth_map

    # Create a mask where the crack is defined (assuming background is zero or near zero)
    crack_mask = resized_crack_depth_map > np.min(resized_crack_depth_map)

    # Where there's a crack, subtract the inverted crack depth map from the glyph depth map
    simulated_crack = np.where(crack_mask, glyph_depth_map - inverted_crack_depth_map, glyph_depth_map)

    # Ensure that no values become negative due to subtraction (no depth less than zero)
    simulated_crack = np.clip(simulated_crack, 0, np.max(glyph_depth_map))

    return simulated_crack


def simulate_missing_parts(cracked_displacement_map, missing_part_probability=0.05, missing_part_size_range=(5, 20)):
    """
    Simulate missing parts in the glyph displacement map.

    :param cracked_displacement_map: 2D numpy array with depth values.
    :param missing_part_probability: The probability of a missing part occurring.
    :param missing_part_size_range: The range of the missing part size in pixels.

    :return: Depth map with simulated missing parts.
    """
    missing_part_map = np.copy(cracked_displacement_map)
    missing_part_mask = np.random.rand(*missing_part_map.shape) < missing_part_probability

    # Generate random missing part sizes
    missing_part_sizes = np.random.randint(*missing_part_size_range, size=missing_part_map.shape)

    # Set the depth values of the missing parts to zero
    missing_part_map[missing_part_mask] = 0

    return missing_part_map


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


def simulate_cracks_in_point_cloud(point_cloud, crack_fraction=0.1, crack_depth=0.5):
    """
    Simulate cracks in the point cloud.
    """

    num_points = len(point_cloud)
    num_crack_points = int(num_points * crack_fraction)
    crack_indices = np.random.choice(num_points, num_crack_points, replace=False)

    point_cloud[crack_indices, 2] *= crack_depth

    return point_cloud


def simulate_missing_parts_in_point_cloud(point_cloud, missing_part_fraction=0.1, missing_part_depth=0.5):
    """
    Simulate missing parts in the point cloud.
    """
    num_points = len(point_cloud)
    num_missing_part_points = int(num_points * missing_part_fraction)
    missing_part_indices = np.random.choice(num_points, num_missing_part_points, replace=False)

    point_cloud[missing_part_indices, 2] *= missing_part_depth

    return point_cloud
