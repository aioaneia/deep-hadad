import utils.cv_file_utils as file_utils
import utils.plot_utils as plot_utils
import utils.cv_convert_utils as conv_utils
import simulation.augmentation_utils as aug_utils

glyph_d_map_path = '../data/test_dataset/Results/test_1.png'

glyph_d_map = file_utils.load_displacement_map(glyph_d_map_path)
glyph_point_cloud = conv_utils.displacement_map_to_point_cloud(glyph_d_map)
glyph_vertices, glyph_faces = conv_utils.displacement_map_to_mesh(glyph_d_map)

# Sharpen the image and add it to the list
glyph_d_map = file_utils.preprocess_displacement_map(glyph_d_map)

# Plot the original image
plot_utils.plot_displacement_map(glyph_d_map, title='Original Displacement Map of G(t)')
plot_utils.plot_displacement_map_geometry_in_3d(glyph_d_map, title='Original 3D Geometry of G(t)')

# Iterate over 10 iterations
for i in range(5):
    # Apply erosion to the image
    augmented_image = aug_utils.augment_preserved_glyph_image(glyph_d_map)

    # Plot the enhanced image
    plot_utils.plot_displacement_map(augmented_image, title='Enhanced Displacement Map')
    # plot_utils.plot_displacement_map_geometry_in_3d(augmented_image, title='Enhanced 3D Geometry')


