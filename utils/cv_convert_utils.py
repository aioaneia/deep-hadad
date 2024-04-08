
import numpy as np


def displacement_map_to_point_cloud(displacement_map):
    """
    Convert a displacement map to a point cloud.

    :param displacement_map: 2D numpy array with depth values.
    :return: x, y, z arrays representing the point cloud coordinates.
    """
    x, y = np.meshgrid(np.arange(displacement_map.shape[1]), np.arange(displacement_map.shape[0]))
    z = displacement_map.flatten()
    x = x.flatten()
    y = y.flatten()

    point_cloud = np.column_stack((x.ravel(), y.ravel(), displacement_map.ravel()))

    return point_cloud
