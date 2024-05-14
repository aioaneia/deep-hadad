
import cv2
import numpy as np

from noise import pnoise2


def surface_roughness(displacement_map, scale=0.1, octaves=4, persistence=0.5, lacunarity=2.0):
    """
    Generate surface roughness on the displacement map using Perlin noise.

    :param displacement_map: 2D numpy array of the displacement values.
    :param scale: Scale factor for the Perlin noise.
    :param octaves: Number of octaves for the noise generation.
    :param persistence: Persistence factor for noise calculation.
    :param lacunarity: Lacunarity factor for noise calculation.
    :return: Modified displacement map with added roughness.
    """
    noise_map = np.zeros_like(displacement_map, dtype=np.float32)

    for i in range(displacement_map.shape[0]):
        for j in range(displacement_map.shape[1]):
            noise_map[i, j] = pnoise2(i * scale,
                                      j * scale,
                                      octaves=octaves,
                                      persistence=persistence,
                                      lacunarity=lacunarity,
                                      repeatx=displacement_map.shape[0],
                                      repeaty=displacement_map.shape[1],
                                      base=0)

    # Normalize and scale noise_map to match the displacement_map values range
    noise_map = cv2.normalize(noise_map, None, displacement_map.min(), displacement_map.max(), cv2.NORM_MINMAX)

    return displacement_map + noise_map


def simulate_erosion_weathering(displacement_map, erosion_size=5, weathering_intensity=0.1, curvature_threshold=10):
    """
    Simulate erosion and weathering effects on the displacement map.

    :param displacement_map: 2D numpy array of the displacement values.
    :param erosion_size: Size of the erosion kernel.
    :param weathering_intensity: Intensity factor for weathering effects.
    :param curvature_threshold: Threshold for detecting high-curvature regions.
    :return: Modified displacement map with erosion and weathering effects.
    """

    # Convert displacement_map to float32 for consistent processing
    displacement_map = displacement_map.astype(np.float32)

    # Applying erosion to simulate weathering
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))

    # Erosion
    eroded_map = cv2.erode(displacement_map, kernel, iterations=1)

    # Apply varied weathering intensity
    # weathered_map = apply_weathering(eroded_map, alpha=weathering_intensity, beta=1.0 - weathering_intensity)

    # Detecting high-curvature regions using Laplacian
    laplacian = cv2.Laplacian(eroded_map, cv2.CV_32F)
    high_curvature_regions = np.abs(laplacian) > curvature_threshold

    # Intensify weathering in high-curvature regions
    intensified_weathering_map = np.where(high_curvature_regions, eroded_map * (1 - weathering_intensity),
                                          eroded_map)

    # Apply Gaussian blur to simulate surface changes due to weathering and erosion
    blurred_map = cv2.GaussianBlur(intensified_weathering_map, (7, 7), 0)

    return blurred_map


def simulate_erosion_weathering_with_canny(displacement_map, erosion_size=5, weathering_intensity=0.1,
                                           canny_threshold1=100, canny_threshold2=200):
    """
    Simulate erosion and weathering effects on the displacement map using Canny edge detection.

    :param displacement_map: 2D numpy array of the displacement values.
    :param erosion_size: Size of the erosion kernel.
    :param weathering_intensity: Intensity factor for weathering effects.
    :param canny_threshold1: Lower threshold for the hysteresis procedure in Canny.
    :param canny_threshold2: Upper threshold for the hysteresis procedure in Canny.
    :return: Modified displacement map with erosion and weathering effects.
    """
    # Applying erosion to simulate basic weathering
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
    eroded_map = cv2.erode(displacement_map, kernel, iterations=1)

    # Convert the image to 8-bit
    eroded_map_8bit = cv2.convertScaleAbs(eroded_map)

    # Apply Canny edge detection
    edges = cv2.Canny(eroded_map_8bit, canny_threshold1, canny_threshold2)

    # Use edges to intensify weathering
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    weathering_mask = np.where(edges_dilated > 0, 1 - weathering_intensity, 1).astype(np.float32)

    weathered_map = eroded_map * weathering_mask
    weathered_map = np.clip(weathered_map, 0, np.max(displacement_map))

    # Apply Gaussian blur to simulate natural erosion and blending of weathered areas
    blurred_map = cv2.GaussianBlur(weathered_map, (7, 7), 0)

    return blurred_map


def simulate_hydraulic_erosion(displacement_map, iterations=50, erosion_strength=0.01, sediment_capacity=0.1):
    """
    Simplified simulation of hydraulic erosion.
    """
    for _ in range(iterations):
        # Calculate the gradient
        dx, dy = np.gradient(displacement_map.astype(float))
        gradient_magnitude = np.sqrt(dx ** 2 + dy ** 2)

        # Normalize the gradient to use as direction of sediment flow
        grad_direction_x = np.divide(dx, gradient_magnitude, out=np.zeros_like(dx), where=gradient_magnitude != 0)
        grad_direction_y = np.divide(dy, gradient_magnitude, out=np.zeros_like(dy), where=gradient_magnitude != 0)

        # Calculate sediment transport
        sediment_transport = gradient_magnitude * erosion_strength
        sediment_transport = np.clip(sediment_transport, 0, sediment_capacity)

        # Apply sediment transport
        for y in range(displacement_map.shape[0]):
            for x in range(displacement_map.shape[1]):
                nx = int(x + grad_direction_x[y, x])
                ny = int(y + grad_direction_y[y, x])
                if 0 <= nx < displacement_map.shape[1] and 0 <= ny < displacement_map.shape[0]:
                    displacement_map[ny, nx] += sediment_transport[y, x]
                    displacement_map[y, x] -= sediment_transport[y, x]

    return np.clip(displacement_map, 0, np.max(displacement_map))


def simulate_thermal_erosion(displacement_map, iterations=30, crack_threshold=0.05):
    """
    Simplified simulation of thermal erosion.
    """
    eroded_displacement_map = displacement_map.copy()

    for _ in range(iterations):
        # Identify potential crack areas (simplified as high gradient areas)
        gradient_magnitude = np.max(np.abs(np.gradient(eroded_displacement_map.astype(float))), axis=0)
        crack_areas = gradient_magnitude > crack_threshold

        # Simulate crack formation by increasing the displacement values
        eroded_displacement_map[crack_areas] += crack_threshold

    # Normalize the eroded displacement map to the range [0, 1]
    eroded_displacement_map = cv2.normalize(
        eroded_displacement_map, None,
        alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return eroded_displacement_map


def simulate_thermal_erosion_2(
        displacement_map,
        iterations=30,
        crack_threshold=0.05,
        smoothing_iterations=5,
        smoothing_kernel_size=(5, 5)):
    """
    Enhanced simulation of thermal erosion.

    :param displacement_map: 2D numpy array of the displacement values.
    :param iterations: Number of iterations to apply thermal erosion.
    :param crack_threshold: Threshold to determine where cracks are likely to form.
    :param smoothing_iterations: Number of times to apply smoothing to reduce isolated spikes.
    :param smoothing_kernel_size: Size of the Gaussian kernel used for smoothing.
    :return: Modified displacement map with thermal erosion effects.
    """
    eroded_displacement_map = displacement_map.copy()

    for _ in range(iterations):
        # Identify potential crack areas (high gradient areas)
        gradient_magnitude = np.max(np.abs(np.gradient(eroded_displacement_map.astype(float))), axis=0)
        crack_areas = gradient_magnitude > crack_threshold

        # Simulate crack formation by increasing the displacement values
        eroded_displacement_map[crack_areas] += crack_threshold

    # Apply Gaussian smoothing to reduce isolated high points
    for _ in range(smoothing_iterations):
        eroded_displacement_map = cv2.GaussianBlur(eroded_displacement_map, smoothing_kernel_size, 0)

    # Normalize the eroded displacement map to the range [0, 1]
    eroded_displacement_map = cv2.normalize(
        eroded_displacement_map, None,
        alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Put the maximul values and the values with minus 0.1 to the median value
    max = np.max(eroded_displacement_map)

    if iterations >= 40:
        eroded_displacement_map[eroded_displacement_map >= 0.5] = np.median(eroded_displacement_map) - 0.05
    else:
        eroded_displacement_map[eroded_displacement_map == max] = np.median(eroded_displacement_map) - 0.05

    return eroded_displacement_map
