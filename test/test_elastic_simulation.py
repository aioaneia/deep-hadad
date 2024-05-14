
import utils.cv_file_utils as file_utils
import utils.plot_utils as plot_utils

import simulation.elastic_simulation as els_simulation


project_path = '../'
glyph_d_map_path = '../data/test_dataset/Real Glyphs/test_3.png'
crack_d_map_dataset_path = '../data/masks_dataset/'


def test_elastic_deformation():
    # Load a well-preserved glyph displacement map
    d_map = file_utils.load_displacement_map(glyph_d_map_path)

    plot_utils.plot_displacement_map(d_map, title='Well Preserved Displacement Map')

    # Simulate elastic deformation of the glyph displacement map
    syn_elastic_deformed_d_map = els_simulation.apply_elastic_transform_2d(d_map, alpha=300, sigma=10)

    # Plot the elastic deformed displacement map
    plot_utils.plot_displacement_map(syn_elastic_deformed_d_map, title='Elastic Deformed Displacement Map')
    plot_utils.plot_displacement_map_geometry_in_3d(syn_elastic_deformed_d_map, title='Elastic Deformed 3D Geometry')

    assert True


if __name__ == "__main__":
    test_elastic_deformation()
