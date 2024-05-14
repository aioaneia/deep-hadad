
import cv2
import numpy as np

import utils.cv_enhancement_utils as en
import utils.cv_file_utils as file_utils
import utils.plot_utils as plot_utils
import utils.cv_convert_utils as conv_utils

glyph_d_map_path = '../data/test_dataset/Results/test_1.png'
crack_d_map_path = '../data/masks_dataset/KAI_x_fracture_1.png'
texture_img = cv2.imread('../data/texture images/KAI_214_texture_4.png')

glyph_d_map = file_utils.load_displacement_map(glyph_d_map_path)
glyph_point_cloud = conv_utils.displacement_map_to_point_cloud(glyph_d_map)
glyph_vertices, glyph_faces = conv_utils.displacement_map_to_mesh(glyph_d_map)

crack_d_map = file_utils.load_displacement_map(crack_d_map_path)

# Sharpen the image and add it to the list
enhanced_image = en.apply_histogram_equalization(glyph_d_map.copy())

# Plot the original image
plot_utils.plot_displacement_map(glyph_d_map, title='Original Displacement Map of G(t)')
plot_utils.plot_displacement_map_geometry_in_3d(glyph_d_map, title='Original 3D Geometry of G(t)')

# Plot the 3D point cloud of the original image
# plot_utils.plot_point_cloud(glyph_point_cloud, title="Point Cloud of G(t)")
# plot_utils.plotly_point_cloud(glyph_point_cloud, title="Point Cloud of G(t)")

# Plot 3D mesh of the original image and the texture
# plot_utils.plot_mesh(glyph_vertices, glyph_faces, title="3D Surface Mesh of G(t)")
# plot_utils.plotly_mesh(glyph_vertices, glyph_faces, title="3D Surface Mesh of G(t)")
plot_utils.plotly_mesh_with_texture(glyph_vertices, glyph_faces, texture_img, title="3D Surface Mesh of G(t)")

# Plot the enhanced image
plot_utils.plot_displacement_map(enhanced_image, title='Enhanced Displacement Map')
plot_utils.plot_displacement_map_geometry_in_3d(enhanced_image, title='Enhanced 3D Geometry')


# Apply Sobel filter to the image
sobel_image = en.apply_sobel(glyph_d_map.copy())
# Plot the Sobel image
plot_utils.plot_displacement_map(sobel_image, title='Sobel Displacement Map')


