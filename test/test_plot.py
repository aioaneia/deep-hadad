import cv2
import numpy as np

import utils.cv_convert_utils as conv_utils
import utils.cv_file_utils as file_utils
import utils.plot_utils as plot_utils
import simulation.erosion_simulation as erosion_simulation
import simulation.image_enhancement as en

syn_abrasion_glyph_d_maps_path = '../data/test_dataset/Sample Abrasion Simulation/'
syn_erosion_glyph_d_maps_path = '../data/test_dataset/Sample Erosion Simulation/'
real_preserved_glyphs_d_maps_path = '../data/test_dataset/Test Case/'
real_damaged_glyphs_d_maps_path = '../data/test_dataset/Real Damaged Glyphs/'

glyph_d_maps = file_utils.load_displacement_maps(
    real_preserved_glyphs_d_maps_path,
    preprocess=True,
    apply_clahe=True
)

texture_img = cv2.imread('../data/texture images/KAI_214_texture_4.png')

for i in range(len(glyph_d_maps)):
    glyph_d_map = glyph_d_maps[i]

    # Convert the displacement map to a point cloud
    glyph_point_cloud = conv_utils.displacement_map_to_point_cloud(glyph_d_map)

    # # Convert the displacement map to a mesh
    # glyph_vertices, glyph_faces = conv_utils.displacement_map_to_mesh(glyph_d_map)


    plot_utils.plot_displacement_map(
        glyph_d_map,
        title=f'Elevation level',
        cmap='gray' # coolwarm
    )

    # The number of iterations for top hat transform to
    max_iterations = 13
    min_iterations = 1

    # iterate backwards from max_iterations to min_iterations
    for i in range(max_iterations, min_iterations, -1):
        syn_eroded_d_map = erosion_simulation.top_hat_transform(
            glyph_d_map,
            kernel_size_range=(14, 18),
            intensity=0.9,
            iterations=i)

        file_utils.save_displacement_map(
            syn_eroded_d_map,
            '../data/test_dataset/Test Case/results/',
            f'erosion_{i}.png',
            normalize=False,
            dtype=np.uint16
        )

        # glyph_point_cloud = conv_utils.displacement_map_to_point_cloud(syn_eroded_d_map)
        # plot_utils.plotly_point_cloud(glyph_point_cloud, title=f"Point Cloud of G({i})")

    # Apply Sobel filter to the image
    edges = en.edge_enhancement(glyph_d_map.copy())

    # Plot the Sobel image
    plot_utils.plot_displacement_map(edges, title='edges', cmap='gray')

    enhanced_image = cv2.addWeighted(glyph_d_map, 0.7, edges, 0.3, 0)

    file_utils.save_displacement_map(
        enhanced_image,
        '../data/test_dataset/Test Case/results/',
        f'enhanced_image.png',
        normalize=False,
        dtype=np.uint16
    )

    plot_utils.plot_displacement_map(enhanced_image, title='enhanced_image')
    glyph_point_cloud = conv_utils.displacement_map_to_point_cloud(enhanced_image)

    plot_utils.plotly_point_cloud(glyph_point_cloud, title=f"Point Cloud of Enhanced Image")

    # Denoising
    denoised = en.denoise_image(enhanced_image)

    # Morphological operations to close gaps in letters
    # kernel = np.ones((3, 3), np.uint8)
    # closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Sharpen the image
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)

    # Normalize back to 0-1 range
    final_image = cv2.normalize(sharpened, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    file_utils.save_displacement_map(
        final_image,
        '../data/test_dataset/Test Case/results/',
        f'final_image.png',
        normalize=False,
        dtype=np.uint16
    )

    glyph_point_cloud = conv_utils.displacement_map_to_point_cloud(final_image)
    plot_utils.plotly_point_cloud(glyph_point_cloud, title=f"Point Cloud of Final Image")

    # # plot_utils.plot_displacement_map_geometry_in_3d(glyph_d_map, title=f'3D Geometry of G({i})')
    #
    # # plot_utils.plot_mesh(glyph_vertices, glyph_faces, title=f"3D Surface Mesh of G({i})")

    # plot_utils.plotly_mesh(glyph_vertices, glyph_faces, title=f"3D Surface Mesh of G({i})")
    #
    # plot_utils.plotly_mesh_with_texture(glyph_vertices, glyph_faces, texture_img, title="3D Surface Mesh of G(t)")



