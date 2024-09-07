import cv2
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


def simulate_crack(glyph_depth_map, crack_depth_map, blur_radius=7, noise_stddev=0.01, min_depth_ratio=0.1):
    """
    Simulate a crack in the glyph depth map by using a crack depth map.

    Assumptions:
        - The background is zero or near zero depth
        - The crack depth map is inverted so that cracks are 'deep'
        - The crack depth map is resized to match the glyph depth map's size
    """

    # Randomly rotate the crack depth map
    angle = np.random.uniform(0, 360)
    rows, cols = crack_depth_map.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    crack_depth_map = cv2.warpAffine(crack_depth_map, rotation_matrix, (cols, rows))

    # Resize the crack depth map to match the glyph depth map's size
    resized_crack_depth_map = cv2.resize(
        crack_depth_map,
        (glyph_depth_map.shape[1], glyph_depth_map.shape[0]),
        interpolation=cv2.INTER_NEAREST)

    # Invert the crack depth map so that cracks are 'deep'
    inverted_crack_depth_map = np.max(resized_crack_depth_map) - resized_crack_depth_map

    # Create a mask where the crack is defined
    crack_mask = resized_crack_depth_map > np.min(resized_crack_depth_map)

    # Add directional bias to crack propagation
    gradient_x, gradient_y = np.gradient(glyph_depth_map)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    directional_bias = np.cos(gradient_direction - angle * np.pi / 180)

    # Add random Gaussian noise to the inverted crack depth map
    noise = np.random.normal(0, noise_stddev, inverted_crack_depth_map.shape)
    noisy_inverted_crack_depth_map = inverted_crack_depth_map + noise
    # modulated_crack = (noisy_inverted_crack_depth_map * (glyph_depth_map / np.max(glyph_depth_map)))
    modulated_crack = (noisy_inverted_crack_depth_map * (glyph_depth_map / np.max(glyph_depth_map)))   # * (1 + directional_bias)
    modulated_crack = np.maximum(modulated_crack, min_depth_ratio * glyph_depth_map)

    # Apply multi-scale Gaussian blur to the crack mask edges
    scales = [3, 5, 7, 9]

    blurred_crack_mask = multi_scale_blur(crack_mask.astype(np.float32), scales)

    # Where there's a crack, subtract the inverted crack depth map from the glyph depth map
    simulated_crack = np.where(
        crack_mask,                         # Where there's a crack
        glyph_depth_map - (modulated_crack * blurred_crack_mask),  # Subtract the crack depth
        glyph_depth_map                     # Otherwise, keep the original depth
    )

    # Ensure that no values become negative due to subtraction (no depth less than zero)
    simulated_crack = np.clip(simulated_crack, 0, np.max(glyph_depth_map))

    return simulated_crack


def apply_mask(glyph_d_map, mask_d_map, blur_radius=7, noise_stddev=0.01, min_depth_ratio=0.1):
    """
    Simulate a crack in the glyph depth map by using a crack depth map.
    """

    # Rotate the mask depth map with 90 or 180 degrees
    angle = np.random.choice([90, 180])
    rows, cols = mask_d_map.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    mask_d_map = cv2.warpAffine(mask_d_map, rotation_matrix, (cols, rows))

    # Resize the crack depth map to match the glyph depth map's size
    resized_crack_depth_map = cv2.resize(
        mask_d_map,
        (glyph_d_map.shape[1], glyph_d_map.shape[0]),
        interpolation=cv2.INTER_NEAREST)

    # Invert the crack depth map so that cracks are 'deep'
    inverted_crack_depth_map = np.max(resized_crack_depth_map) - resized_crack_depth_map

    # Create a mask where the crack is defined
    crack_mask = resized_crack_depth_map > np.min(resized_crack_depth_map)

    # Add random Gaussian noise to the inverted crack depth map
    noise = np.random.normal(0, noise_stddev, inverted_crack_depth_map.shape)
    noisy_inverted_crack_depth_map = inverted_crack_depth_map + noise
    modulated_crack = (noisy_inverted_crack_depth_map * (glyph_d_map / np.max(glyph_d_map)))   # * (1 + directional_bias)
    modulated_crack = np.maximum(modulated_crack, min_depth_ratio * glyph_d_map)

    # Apply multi-scale Gaussian blur to the crack mask edges
    scales = [3, 5, 7, 9]

    blurred_crack_mask = multi_scale_blur(crack_mask.astype(np.float32), scales)

    # Where there's a crack, subtract the inverted crack depth map from the glyph depth map
    simulated_crack = np.where(
        crack_mask,                         # Where there's a crack
        glyph_d_map - (modulated_crack * blurred_crack_mask),  # Subtract the crack depth
        glyph_d_map                     # Otherwise, keep the original depth
    )

    # Ensure that no values become negative due to subtraction (no depth less than zero)
    simulated_crack = np.clip(simulated_crack, 0, np.max(glyph_d_map))

    return simulated_crack


def multi_scale_blur(mask, scales):
    blurred_mask = np.zeros_like(mask, dtype=np.float32)

    for scale in scales:
        blurred_mask += cv2.GaussianBlur(mask, (scale, scale), 0)

    return blurred_mask / len(scales)


# def simulate_crack_with_poisson_blending(glyph_depth_map, crack_depth_map, blending_factor=0.5):
#     """Simulate a crack on a glyph depth map using Poisson blending."""
#     # Resize the crack map to match the glyph map
#     crack_resized = cv2.resize(crack_depth_map, (glyph_depth_map.shape[1], glyph_depth_map.shape[0]),
#                                interpolation=cv2.INTER_LINEAR)
#
#     # Normalize and scale the crack map inversely to simulate a crack
#     crack_inverted = np.max(crack_resized) - crack_resized
#     crack_normalized = crack_inverted / np.max(crack_inverted) * blending_factor
#
#     # Calculate gradients
#     glyph_grad_x, glyph_grad_y = np.gradient(glyph_depth_map)
#     crack_grad_x, crack_grad_y = np.gradient(crack_normalized)
#
#     # Combine gradients
#     combined_x = glyph_grad_x + crack_grad_x
#     combined_y = glyph_grad_y + crack_grad_y
#
#     # Setup Poisson problem
#     n = glyph_depth_map.size
#     rows, cols = glyph_depth_map.shape
#     A = lil_matrix((n, n))
#     b = np.zeros(n)
#
#     def index(i, j):
#         return i * cols + j
#
#     # Fill the matrix A
#     for i in range(rows):
#         for j in range(cols):
#             idx = index(i, j)
#             A[idx, idx] = 4
#             if i > 0:
#                 A[idx, index(i - 1, j)] = -1
#             if i < rows - 1:
#                 A[idx, index(i + 1, j)] = -1
#             if j > 0:
#                 A[idx, index(i, j - 1)] = -1
#             if j < cols - 1:
#                 A[idx, index(i, j + 1)] = -1
#             b[idx] = combined_x[i, j] + combined_y[i, j]
#
#     # Solve the Poisson equation
#     solution = spsolve(csr_matrix(A), b)
#
#     # Reshape the solution to the image shape
#     result = solution.reshape((rows, cols))
#
#     return result


# def simulate_cracks_in_point_cloud(point_cloud, crack_fraction=0.1, crack_depth=0.5):
#     """
#     Simulate cracks in the point cloud.
#     """
#
#     num_points = len(point_cloud)
#     num_crack_points = int(num_points * crack_fraction)
#     crack_indices = np.random.choice(num_points, num_crack_points, replace=False)
#
#     point_cloud[crack_indices, 2] *= crack_depth
#
#     return point_cloud
