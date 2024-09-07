import numpy as np

from scipy.ndimage import gaussian_filter, map_coordinates


def apply_elastic_transform_2d(image, alpha, sigma, random_state=None):
    """
    Apply an elastic deformation on a 2D image.
    The function uses Gaussian filters to smooth random displacement fields.

    Parameters:
    image (numpy.ndarray): The input 2D image.
    alpha (float): Intensity of deformation. Larger values result in more distortion.
    sigma (float): Smoothing factor for Gaussian filter (standard deviation). Larger values result in smoother fields.
    random_state (numpy.random.RandomState, optional): Random state for reproducibility.

    Returns:
    numpy.ndarray: Distorted 2D image.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    new_random_state = (random_state.rand(*shape) * 2 - 1)

    # Generate random displacement fields
    dx = gaussian_filter(new_random_state, sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(new_random_state, sigma, mode="constant", cval=0) * alpha

    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    # Apply displacement fields to the coordinates
    # indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # Map coordinates from input image to distorted image
    distorted_image = map_coordinates(image, indices, order=1, mode='reflect')

    return distorted_image.reshape(shape)


# def elastic_transform_3d(volume, alpha, sigma, random_state=None):
#     """
#     Apply an elastic deformation on a 3D volume. The function uses Gaussian filters
#     to smooth random displacement fields.
#
#     Parameters:
#         volume (numpy.ndarray): The input 3D volume.
#         alpha (float): Intensity of deformation.
#         sigma (float): Smoothing factor for Gaussian filter.
#         random_state (numpy.random.RandomState, optional): Random state for reproducibility.
#
#     Returns:
#         numpy.ndarray: Distorted volume.
#     """
#     if random_state is None:
#         random_state = np.random.RandomState(None)
#
#     shape = volume.shape
#     dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
#     dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
#     dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
#
#     x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]), indexing='ij')
#     indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))
#
#     distorted_volume = map_coordinates(volume, indices, order=1, mode='reflect')
#
#     return distorted_volume.reshape(shape)
