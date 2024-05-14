import utils.cv_file_utils as file_utils
import utils.plot_utils as plot_utils

project_path = '../'

# Paths for the test glyph displacement map and crack displacement map
glyph_d_map_path = project_path + 'data/test_dataset/Real Glyphs/test_1.png'


def test_preprocessing():
    # Load a well-preserved glyph displacement map
    d_map = file_utils.load_displacement_map(glyph_d_map_path)

    # Preprocess the displacement map
    preprocessed_d_map = file_utils.preprocess_displacement_map(d_map)

    # Plot the original displacement map
    plot_utils.plot_displacement_map(d_map, title='Original Glyph Displacement Map', cmap='gray')

    # Plot the displacement map
    plot_utils.plot_displacement_map(preprocessed_d_map, title='Glyph Displacement Map', cmap='gray')

    # plot_utils.plot_heatmap_from_displacement_map(d_map, title='Glyph Heatmap')
    plot_utils.plot_displacement_map_geometry_in_3d(d_map, title='Glyph 3D Geometry')

    # plot_utils.plot_heatmap_from_displacement_map(d_map, title='Glyph Heatmap')
    plot_utils.plot_displacement_map_geometry_in_3d(preprocessed_d_map, title='Glyph 3D Geometry')

    assert True


####################################################################################################
# Main function
# - Load a good glyph image
####################################################################################################
if __name__ == "__main__":
    # Test the preprocessing of displacement maps
    test_preprocessing()

