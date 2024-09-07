
import cv2
import numpy as np

from noise import pnoise2


def surface_roughness(displacement_map, scale=0.1, octaves=4, persistence=0.5, lacunarity=2.0):
    """
    Generate surface roughness on the displacement map using Perlin noise.
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


def layer_separation_simulation(displacement_map, num_layers=3, separation_probability=0.2):
    # Convert to float32 for processing
    result = displacement_map.astype(np.float32)

    original_max = np.max(result)
    original_min = np.min(result)

    # Normalize to [0, 1] range
    result = (result - original_min) / (original_max - original_min)

    for _ in range(num_layers):
        mask = np.random.rand(*displacement_map.shape) < separation_probability
        depth = np.random.uniform(0.2, 0.8)
        result[mask] *= depth

    # Smooth the edges of separated layers
    result = cv2.GaussianBlur(result, (5, 5), 0)

    # Scale back to original range
    result = result * (original_max - original_min) + original_min

    # Clip to ensure we're within the original range
    result = np.clip(result, original_min, original_max)

    # Convert back to original dtype
    if displacement_map.dtype == np.uint8:
        result = result.round().astype(np.uint8)

    return result


def patina_formation(depth_map, thickness=0.5, coverage=0.8):
    patina = np.random.rand(*depth_map.shape) < coverage
    result = depth_map.astype(np.float32)
    result[patina] += thickness * np.max(depth_map)
    return np.clip(result, 0, np.max(result))


def water_erosion_channels(depth_map, num_channels=5, depth=0.2):
    result = depth_map.astype(np.float32)

    for _ in range(num_channels):
        start = np.random.randint(0, depth_map.shape[1])
        path = np.zeros(depth_map.shape, dtype=bool)
        current = start
        for i in range(depth_map.shape[0]):
            path[i, current] = True
            current += np.random.randint(-1, 2)
            current = np.clip(current, 0, depth_map.shape[1] - 1)

        result[path] -= depth * np.max(depth_map)

    return np.clip(result, 0, np.max(depth_map))


def biological_growth(depth_map, coverage=0.9, thickness=0.2):
    growth = np.random.rand(*depth_map.shape) < coverage
    growth = cv2.GaussianBlur(growth.astype(np.float32), (5, 5), 0)
    result = depth_map.astype(np.float32)
    result += growth * thickness * np.max(depth_map)
    return np.clip(result, 0, np.max(result))