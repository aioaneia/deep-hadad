
import glob
import os
import matplotlib.pyplot as plt
import torch

from torchvision.transforms import ToPILImage, Compose, ToTensor

import utils.cv_file_utils as file_utils

from models.Spade2Generator import Spade2Generator


# PyTorch version
print("PyTorch version: " + torch.__version__)

# Constants
PROJECT_PATH                       = '../'
synthetic_damage_test_dataset_path = PROJECT_PATH + "data/test_dataset/Synthetic Damaged Glyphs"
real_damage_test_dataset_path      = PROJECT_PATH + "data/test_dataset/Real Damaged Glyphs/"

IMAGE_EXTENSIONS = [".png", ".jpg", ".tif"]

MODEL_PATH = PROJECT_PATH + 'trained_models/'

MODEL_NAMES = [
    'dh_model_ep_1_l100.00_s10.00_m5.00_g10.00_t0.10_f5.00_a0.40.pth',
    'dh_model_ep_3_l100.00_s10.00_m5.00_g10.00_t0.10_f5.00_a0.40.pth',
    'dh_model_ep_9_l100.00_s10.00_m5.00_g10.00_t0.10_f5.00_a0.40.pth',
    "dh_model_ep_11_l100.00_s10.00_m5.00_g10.00_t0.10_f5.00_a0.40.pth"
]

transform = Compose([
    ToTensor(),
])

to_pil = ToPILImage()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint)

    return model.eval()


def load_dh_generator():
    """ Loads the generator for the DHadad model """
    return Spade2Generator(
        input_nc=1,
        output_nc=1,
        label_nc=1,
        ngf=64,
        n_downsampling=3,
        n_blocks=9
    ).to(device)


def load_image_pairs(path):
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory")

    image_pairs = []

    all_files = sorted(
        glob.glob(os.path.join(path, '*.png')) +
        glob.glob(os.path.join(path, '*.tif'))
    )

    print(f"Number of files: {len(all_files)}")

    # Group files into pairs
    for i in range(0, len(all_files), 2):
        if i + 1 < len(all_files):
            damage_file = all_files[i]
            ground_file = all_files[i + 1]

            ground_image = file_utils.load_displacement_map(
                ground_file,
                preprocess=True,
                resize=False,
                apply_clahe=False
            )

            damage_image = file_utils.load_displacement_map(
                damage_file,
                preprocess=True,
                resize=False,
                apply_clahe=False
            )

            image_pairs.append({
                'ground_truth': ground_image,
                'damage_image': damage_image
            })

    return image_pairs


def generate_restored_image(generator, test_image_tensor, invert_pixel_values=True):
    with torch.no_grad():
        # Add a batch dimension and move to the GPU if needed
        broken_image = test_image_tensor.unsqueeze(0).to(device)
        segmap = torch.zeros_like(broken_image)

        # Generate the restored image and remove the batch dimension
        restored_image = generator(broken_image, segmap).squeeze(0).cpu()

        # Normalize the image to the range [0, 1]
        restored_image = (restored_image - restored_image.min()) / (restored_image.max() - restored_image.min())

    if invert_pixel_values == True:
        restored_image = 1 - restored_image

    return restored_image


def generate_restored_images(generators, image_pairs, image_size=(256, 256)):
    table_images = []

    for pair in image_pairs:
        # Resize displacement maps while preserving aspect ratio
        resized_ground = file_utils.resize_and_pad_depth_map(pair['ground_truth'], target_size=image_size)
        resized_damage = file_utils.resize_and_pad_depth_map(pair['damage_image'], target_size=image_size)

        row = {
            'ground_truth': resized_ground,
            'damage_image': resized_damage,
        }

        # Transform the resized_map to a tensor
        damage_tensor = transform(resized_damage)

        # Generate the restored image for each generator
        for j, generator in enumerate(generators):
            restored_image = generate_restored_image(generator, damage_tensor, invert_pixel_values=False)

            row[f'restored_image_{j}'] = to_pil(restored_image)

        table_images.append(row)

    return table_images


def plot_qualitative_table(image_table, cmap='gray'):
    num_images = len(image_table)

    fig, axes = plt.subplots(
        num_images,
        len(MODEL_NAMES) + 2,
        figsize=(15, 5 * num_images)
    )

    # Plot the images from the table
    for i, table_image in enumerate(image_table):
        # Show ground truth image
        axes[i, 0].imshow(table_image['ground_truth'], cmap=cmap)
        axes[i, 0].set_title('Ground Truth Image')

        # Show restored image
        axes[i, 1].imshow(table_image['damage_image'], cmap=cmap)
        axes[i, 1].set_title('Damage Image')

        # Show restored images
        for j in range(len(MODEL_NAMES)):
            axes[i, j + 2].imshow(table_image[f'restored_image_{j}'], cmap=cmap)
            axes[i, j + 2].set_title(f'Restored Image {j}')

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


    for i, table_image in enumerate(image_table):
        fig, axes = plt.subplots(
            1,
            len(MODEL_NAMES) + 2,
            figsize=(30, 30)
        )

        axes[0].imshow(table_image['ground_truth'], cmap=cmap)
        axes[0].set_title('Ground Truth Image')

        axes[1].imshow(table_image['damage_image'], cmap=cmap)
        axes[1].set_title('Damage Image')

        for j in range(len(MODEL_NAMES)):
            axes[j + 2].imshow(table_image[f'restored_image_{j}'], cmap=cmap)
            axes[j + 2].set_title(f'Restored Image {j}')

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Load the DeepHadad generator
    generators = [load_model(load_dh_generator(), os.path.join(MODEL_PATH, name)) for name in MODEL_NAMES]

    # Load the image pairs
    image_pairs = load_image_pairs(synthetic_damage_test_dataset_path)

    # Generate the restored images
    table_images = generate_restored_images(generators, image_pairs)

    # Plot the qualitative table
    plot_qualitative_table(table_images)
