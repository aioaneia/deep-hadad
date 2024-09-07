
import numpy as np


# def displacement_map_to_point_cloud(displacement_map):
#     """
#     Convert a displacement map to a point cloud.
#     """
#     x, y = np.meshgrid(np.arange(displacement_map.shape[1]), np.arange(displacement_map.shape[0]))
#
#     z = displacement_map.flatten()
#
#     x = x.flatten()
#     y = y.flatten()
#
#     point_cloud = np.column_stack((x.ravel(), y.ravel(), displacement_map.ravel()))
#
#     return point_cloud


def displacement_map_to_point_cloud(displacement_map):
    """
    Convert a displacement map to a point cloud, correcting for the mirroring effect.

    :param displacement_map: 2D numpy array with depth values.
    :return: point_cloud: Nx3 numpy array representing the point cloud coordinates.
    """
    height, width = displacement_map.shape

    # Create a meshgrid of coordinates
    y, x = np.mgrid[0:height, 0:width]

    # Flip the y-coordinates to match the image orientation
    y = height - 1 - y

    # Flatten the arrays
    x = x.flatten()
    y = y.flatten()
    z = displacement_map.flatten()

    # Stack the coordinates
    point_cloud = np.column_stack((x, y, z))

    return point_cloud


def displacement_map_to_mesh(displacement_map):
    """
    Convert a displacement map to a mesh representation (vertices and faces).

    Parameters:
        displacement_map (np.array): 2D array with depth values.

    Returns:
        vertices (np.array): Array of shape (n_points, 3) containing x, y, z coordinates of the vertices.
        faces (np.array): Array of shape (n_faces, 3) containing indices of vertices that form each triangular face.
    """
    # Create meshgrid arrays for x and y coordinates
    x, y = np.meshgrid(np.linspace(0, 1, displacement_map.shape[1]),
                       np.linspace(0, 1, displacement_map.shape[0]))

    # Flatten the x, y and displacement map arrays to create a list of coordinates
    z = displacement_map.flatten()
    x = x.flatten()
    y = y.flatten()

    # Combine x, y, and z into a single array of vertices
    vertices = np.column_stack((x, y, z))

    # Generate the faces (triangles) connecting vertices
    faces = []
    cols = displacement_map.shape[1]
    for i in range(displacement_map.shape[0] - 1):
        for j in range(displacement_map.shape[1] - 1):
            # Top left triangle in square grid
            faces.append([i * cols + j, i * cols + (j + 1), (i + 1) * cols + j])
            # Bottom right triangle in square grid
            faces.append([(i + 1) * cols + j, i * cols + (j + 1), (i + 1) * cols + (j + 1)])

    faces = np.array(faces)

    return vertices, faces
