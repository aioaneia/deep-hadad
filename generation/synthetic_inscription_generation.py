import os
import random
import numpy as np
import cv2

def load_and_normalize_letter_depth_maps(folder_path, target_height=256):
    letter_maps = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            letter = filename.split('.')[0]
            img_path = os.path.join(folder_path, filename)
            depth_map = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            depth_map = preprocess_displacement_map(depth_map)
            resized_map = resize_depth_map(depth_map, target_height=target_height)
            letter_maps[letter] = resized_map
    return letter_maps

def preprocess_displacement_map(d_map, apply_clahe=False):
    d_map = cv2.medianBlur(d_map, 5)
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        d_map = clahe.apply(d_map)
    d_map = cv2.normalize(d_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return d_map

def resize_depth_map(depth_map, target_height=256):
    aspect_ratio = depth_map.shape[1] / depth_map.shape[0]
    new_width = int(target_height * aspect_ratio)
    return cv2.resize(depth_map, (new_width, target_height), interpolation=cv2.INTER_LINEAR)


def generate_inscription_patch(letter_maps, letters_per_line, num_lines, line_spacing_factor=1.2):
    available_letters = list(letter_maps.keys())
    max_letter_width = max(letter_map.shape[1] for letter_map in letter_maps.values())
    letter_height = list(letter_maps.values())[0].shape[0]

    canvas_width = int(max_letter_width * letters_per_line * 1.1)
    canvas_height = int(letter_height * num_lines * line_spacing_factor)

    # Create a lighter textured background
    background = create_textured_background(canvas_width, canvas_height)

    y_offset = int(letter_height * 0.2)
    for _ in range(num_lines):
        x_offset = int(max_letter_width * 0.1)
        for _ in range(letters_per_line):
            letter = random.choice(available_letters)
            letter_map = letter_maps[letter].copy()

            # Ensure letter_map is in the range [0, 255]
            letter_map = cv2.normalize(letter_map, None, 0, 255, cv2.NORM_MINMAX)

            # Add slight random variations
            angle = random.uniform(-1, 1)
            scale = random.uniform(0.95, 1.05)
            M = cv2.getRotationMatrix2D((letter_map.shape[1] // 2, letter_map.shape[0] // 2), angle, scale)
            rotated_letter = cv2.warpAffine(letter_map, M, letter_map.shape[:2], borderMode=cv2.BORDER_REPLICATE)

            h, w = rotated_letter.shape
            if x_offset + w > canvas_width:
                break
            if y_offset + h > canvas_height:
                break

            # Blend the letter with the background, making letters darker
            alpha = rotated_letter / 255.0
            letter_color = 50  # Darker color for letters
            background[y_offset:y_offset + h, x_offset:x_offset + w] = \
                (1 - alpha) * background[y_offset:y_offset + h, x_offset:x_offset + w] + \
                alpha * letter_color

            x_offset += w + random.randint(1, 5)

        y_offset += int(letter_height * line_spacing_factor)
        if y_offset > canvas_height:
            break

    return add_realistic_wear(background)


def create_textured_background(width, height):
    # Create a lighter noisy background
    background = np.random.normal(220, 15, (height, width)).astype(np.float32)
    background = cv2.GaussianBlur(background, (7, 7), 0)
    return np.clip(background, 180, 255)  # Ensure background is light


def add_realistic_wear(image, wear_factor=0.03):
    worn_image = image.copy()
    noise = np.random.normal(0, wear_factor * 255, image.shape)
    worn_image += noise
    erosion_kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(worn_image, erosion_kernel, iterations=1)
    worn_image = cv2.addWeighted(worn_image, 0.8, eroded, 0.2, 0)
    return np.clip(worn_image, 0, 255).astype(np.uint8)

def generate_multiple_patches(letter_maps, num_patches, min_letters_per_line=5, max_letters_per_line=10, min_lines=3, max_lines=10):
    patches = []
    for _ in range(num_patches):
        letters_per_line = random.randint(min_letters_per_line, max_letters_per_line)
        num_lines = random.randint(min_lines, max_lines)
        patch = generate_inscription_patch(letter_maps, letters_per_line, num_lines)
        patches.append(patch)
    return patches

# Usage
folder_path = '../data/glyphs_dataset/preserved_glyphs/dm_panamuwa_script'
letter_maps = load_and_normalize_letter_depth_maps(folder_path)
generated_patches = generate_multiple_patches(letter_maps, num_patches=10)

# Save generated patches
for i, patch in enumerate(generated_patches):
    cv2.imwrite(f'../data/syn_inscriptions/generated_inscription_{i}.png', (patch * 255).astype(np.uint8))