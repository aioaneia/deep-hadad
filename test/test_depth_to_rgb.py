
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging


def apply_depth_to_rgb(rgb_image, depth_map):
    """
    Apply the enhanced depth map to the RGB image as a shading layer.

    Args:
    - rgb_image: Original RGB image as a NumPy array.
    - depth_map: Enhanced depth map as a NumPy array.

    Returns:
    - shaded_rgb: RGB image with depth shading applied.
    """

    # Normalize depth map
    depth_map_normalized = cv2.normalize(depth_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Apply depth map as shading to RGB image
    shaded_rgb = rgb_image * depth_map_normalized[..., np.newaxis]

    return shaded_rgb


def test_depth_to_rgb(rgb_image_path, depth_map_path):
    # Load your RGB image and depth map
    rgb_image = cv2.imread(rgb_image_path, cv2.COLOR_BGR2RGB)  # Replace with your image path
    # rgb_image = Image.open(rgb_image_path).convert("RGB")
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)  # Replace with your depth map path

    # Enhance the depth map
    # depth_map = augment_image(depth_map, pipeline=enhacement_augmentation_pipeline)

    # rgb_image_np = np.array(rgb_image)

    # Resize depth map to match the size of the RGB image
    depth_map = cv2.resize(depth_map, (rgb_image.shape[1], rgb_image.shape[0]))

    # Apply the depth map to the RGB image
    shaded_rgb_image = apply_depth_to_rgb(rgb_image, depth_map)

    # Save or display the result
    # cv2.imwrite('shaded_rgb_image.jpg', shaded_rgb_image)  # Save the shaded image

    # Dispay results
    display_images([rgb_image, depth_map, shaded_rgb_image],
                   titles=["RGB", "Depth", "Shaded RGB"],
                   num_cols=3, img_size=(100, 100), cmap='color')

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    # Show original image
    # axes[0].imshow(rgb_image)
    # axes[0].set_title('Original Image')
    # axes[0].axis('off')

    # # Show estimated image
    # axes[1].imshow(depth_map, cmap='gray')
    # axes[1].set_title('Estimated Image')
    # axes[1].axis('off')

    # # Show restored image
    # axes[2].imshow(shaded_rgb_image)
    # axes[2].set_title('Restored Image')
    # axes[2].axis('off')

    # plt.tight_layout()
    # plt.show()


####################################################################################################
# Main
# - Load displacement maps
# - Generate synthetic displacement maps
# - Validate the generated data
####################################################################################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Read parameters from a configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Get the path to the test dataset directory
    rgb_images = config['DEFAULT']['RGB_TEST_DATASET_PATH']
    ground_d_m_path = config['DEFAULT']['GROUND_TEST_DATASET_PATH']
    est_d_m_path = config['DEFAULT']['EST_TEST_DATASET_PATH']
    dh_d_m_path = config['DEFAULT']['DH_TEST_DATASET_PATH']

    # Test displacement maps generation
    # load_pairs_of_est_ground(ground_d_m_path, est_d_m_path)

    # test depth to rgb
    test_depth_to_rgb(rgb_images + 'KAI_214_L10-11_2.png', dh_d_m_path + 'KAI_214_L10-11_2.png')

    print("Prevalidation successful.")