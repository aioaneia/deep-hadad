import cv2

import utils.cv_convert_utils as conv_utils
import utils.cv_file_utils as file_utils
import utils.plot_utils as plot_utils

syn_abrasion_glyph_d_maps_path = '../data/test_dataset/Sample Abrasion Simulation/'
syn_erosion_glyph_d_maps_path = '../data/test_dataset/Sample Erosion Simulation/'
real_preserved_glyphs_d_maps_path = '../data/test_dataset/Real Preserved Glyphs/'
real_damaged_glyphs_d_maps_path = '../data/test_dataset/Real Damaged Glyphs/'

glyph_d_maps = file_utils.load_displacement_maps(real_preserved_glyphs_d_maps_path, preprocess=True, apply_clahe=True)

texture_img = cv2.imread('../data/texture images/KAI_214_texture_4.png')

for i in range(len(glyph_d_maps)):
    glyph_d_map = glyph_d_maps[i]

    # Convert the displacement map to a point cloud
    glyph_point_cloud = conv_utils.displacement_map_to_point_cloud(glyph_d_map)

    # Convert the displacement map to a mesh
    glyph_vertices, glyph_faces = conv_utils.displacement_map_to_mesh(glyph_d_map)

    plot_utils.plot_displacement_map(glyph_d_map, title=f'Displacement Map of G({i})', cmap='gray')

    plot_utils.plot_displacement_map(glyph_d_map, title=f'Displacement Map of G({i})', cmap='coolwarm')

    # plot_utils.plot_displacement_map_geometry_in_3d(glyph_d_map, title=f'3D Geometry of G({i})')

    # plot_utils.plot_point_cloud(glyph_point_cloud, title=f"Point Cloud of G({i})")
    plot_utils.plotly_point_cloud(glyph_point_cloud, title=f"Point Cloud of G({i})")

    # plot_utils.plot_mesh(glyph_vertices, glyph_faces, title=f"3D Surface Mesh of G({i})")
    plot_utils.plotly_mesh(glyph_vertices, glyph_faces, title=f"3D Surface Mesh of G({i})")

    plot_utils.plotly_mesh_with_texture(glyph_vertices, glyph_faces, texture_img, title="3D Surface Mesh of G(t)")



