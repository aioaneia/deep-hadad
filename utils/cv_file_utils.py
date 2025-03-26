import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import logging
import torch
import OpenEXR
import Imath
import imageio


IMAGE_EXTENSIONS = [".png", ".jpg", "JPG", ".jpeg", ".tif'", ".tiff", ".exr", ".bmp"]


def load_displacement_maps_from_directory(path, preprocess=False, resize=False):
    """Load displacement maps from the specified directory"""
    displacement_maps = []
    for subdir, dirs, files in os.walk(path):
        for dir in sorted(dirs):
            full_dir_path = os.path.join(subdir, dir)
            displacement_maps += load_displacement_maps(full_dir_path, preprocess=preprocess,resize=resize)
    print(f'Number of Displacement Maps: {len(displacement_maps)}')
    return displacement_maps


def load_crack_displacement_maps_from_directory(path, preprocess=False):
    """Load crack displacement maps from the specified directory"""
    crack_d_map_paths = get_image_paths(path)
    crack_d_maps = []
    for crack_d_map_path in crack_d_map_paths:
        crack_d_map = load_displacement_map(crack_d_map_path, preprocess=preprocess)
        crack_d_maps.append(crack_d_map)
    print(f'Number of Crack Displacement Maps: {len(crack_d_maps)}')
    return crack_d_maps


def load_displacement_maps(path, preprocess=False, resize=False, apply_clahe=False):
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory")

    map_paths = sorted(glob.glob(os.path.join(path, '*.png')) +
                       glob.glob(os.path.join(path, '*.exr')) +
                       glob.glob(os.path.join(path, '*.tif')),
                       key=lambda x: os.path.basename(x))

    displacement_maps = []
    for map_path in tqdm(map_paths, desc="Loading displacement maps"):
        displacement_map = load_displacement_map(map_path, preprocess=preprocess, resize=resize,
                                                 apply_clahe=apply_clahe)
        displacement_maps.append(displacement_map)
        logging.info(f"File: {map_path} loaded successfully.")
    return displacement_maps


def load_displacement_map(d_map_path, preprocess=False, resize=False, apply_clahe=False, target_size=(256, 256)):
    """Load a displacement map with appropriate handling based on file type"""
    # Determine file type
    file_ext = os.path.splitext(d_map_path)[1].lower()

    if file_ext == '.exr':
        d_map = load_exr_displacement_map(d_map_path)
        # d_map = pyexr.read(d_map_path).squeeze()  # Shape: [H, W]
        # d_map = d_map.astype(np.float32)
    else:
        # Standard loading for other formats
        d_map = cv2.imread(d_map_path, cv2.IMREAD_UNCHANGED)

        # Handle color images
        if d_map is not None and d_map.ndim == 3:
            d_map = cv2.cvtColor(d_map, cv2.COLOR_BGR2GRAY)

    if d_map is None:
        logging.warning(f"Could not load {d_map_path}")
        return None

    # Normalize based on data type
    if d_map.dtype != np.float32:
        if d_map.dtype == np.uint8:
            d_map = d_map.astype(np.float32) / 255.0
        elif d_map.dtype == np.uint16:
            d_map = d_map.astype(np.float32) / 65535.0
        else:
            # For other types, normalize to [0,1]
            min_val = d_map.min()
            max_val = d_map.max()
            if max_val > min_val:
                d_map = (d_map.astype(np.float32) - min_val) / (max_val - min_val)
            else:
                d_map = d_map.astype(np.float32)

    if preprocess:
        d_map = preprocess_displacement_map(d_map, apply_clahe=apply_clahe)

    if resize:
        d_map = resize_and_pad_depth_map(d_map, target_size=target_size)

    # # Normalizing the pixel values to the range [0, 1]
    # d_map = cv2.normalize(d_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return d_map


def load_exr_displacement_map(exr_path):
    """Load an OpenEXR file as a displacement map with fallback methods"""
    try:
        # Try OpenCV first
        d_map = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)

        if d_map is None or d_map.size == 0:
            # If OpenCV fails, try OpenEXR library
            try:
                exr_file = OpenEXR.InputFile(exr_path)
                dw = exr_file.header()['dataWindow']
                size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

                # Try to get the channel that contains displacement data
                channel_names = exr_file.header()['channels'].keys()
                if 'Y' in channel_names:
                    channel = 'Y'  # Luminance channel
                elif 'R' in channel_names:
                    channel = 'R'  # Red channel as fallback
                else:
                    channel = list(channel_names)[0]  # First available channel

                FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                channel_str = exr_file.channel(channel, FLOAT)

                # Convert to numpy array
                channel_arr = np.frombuffer(channel_str, dtype=np.float32)
                d_map = channel_arr.reshape(size[1], size[0])

            except ImportError:
                # If OpenEXR is not available, try imageio
                d_map = imageio.imread(exr_path)
                # Extract appropriate channel if multi-channel
                if d_map.ndim > 2:
                    d_map = d_map[:, :, 0]  # First channel

        # If the map has multiple channels, extract the first one
        if d_map is not None and d_map.ndim > 2:
            d_map = d_map[:, :, 0]

        # Convert to float32 if needed
        if d_map is not None and d_map.dtype != np.float32:
            d_map = d_map.astype(np.float32)

        return d_map

    except Exception as e:
        logging.error(f"Error loading EXR file {exr_path}: {str(e)}")
        return None


def preprocess_displacement_map(d_map, apply_clahe=False):
    """Preprocess displacement map with displacement-specific considerations"""
    # Ensure float32 format
    if d_map.dtype != np.float32:
        d_map = d_map.astype(np.float32)

    # Normalize to [0,1] if needed
    if d_map.min() < 0 or d_map.max() > 1:
        d_map = (d_map - d_map.min()) / (d_map.max() - d_map.min() + 1e-8)

    # Use bilateral filter instead of median for edge-preserving noise reduction
    # This preserves glyph edges while reducing noise
    d_map = cv2.bilateralFilter(d_map, d=5, sigmaColor=0.1, sigmaSpace=5)
    # d_map = cv2.medianBlur(d_map, 5)

    # Apply CLAHE if requested (with special handling for float values)
    if apply_clahe:
        # Store original min/max values
        orig_min = d_map.min()
        orig_range = d_map.max() - orig_min

        # Convert to 8-bit for CLAHE
        d_map = cv2.normalize(d_map, None, 0, 255, cv2.NORM_MINMAX)
        d_map = d_map.astype(np.uint8)

        # Use a gentler CLAHE setting for archaeological features
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        d_map_8bit = clahe.apply(d_map)

        # Convert back to float32
        d_map = d_map_8bit.astype(np.float32) / 255.0
        d_map = d_map * orig_range + orig_min

    return d_map


def resize_and_pad_depth_map(depth_map, target_size=(256, 256)):
    h, w = depth_map.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize using INTER_LINEAR for depth maps
    interpolation = cv2.INTER_LINEAR
    depth_map_resized = cv2.resize(depth_map, (new_w, new_h), interpolation=interpolation)

    # Calculate padding
    top = (target_size[0] - new_h) // 2
    bottom = target_size[0] - new_h - top
    left = (target_size[1] - new_w) // 2
    right = target_size[1] - new_w - left

    # Pad with edge values instead of zeros
    depth_map_padded = cv2.copyMakeBorder(
        depth_map_resized,
        top, bottom, left, right,
        cv2.BORDER_REPLICATE
    )

    return depth_map_padded


def transform_displacement_map_to_tensor(image):
    """Converts a displacement map image into a torch tensor, normalizing the data"""
    image_tensor = torch.from_numpy(image).float()

    # Normalize the image tensor to [0, 1] if it's not already
    image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())

    return image_tensor.unsqueeze(0)


def save_paired_images(input_d_map, target_d_map, input_path, target_path, set_index, pair_index):
    """
    Save a pair of depth images (x and y) to the specified paths.
    """
    # Construct unique filenames for the x and y images
    filename_input = f"i_pair_{set_index}_{pair_index}.png"
    filename_target = f"t_pair_{set_index}_{pair_index}.png"

    # Rescale the images from [0, 1] to [0, 255]
    input_d_map_rescaled = (input_d_map * 255).astype(np.uint8)
    target_d_map_rescaled = (target_d_map * 255).astype(np.uint8)

    # Save the images
    cv2.imwrite(os.path.join(input_path, filename_input), input_d_map_rescaled)
    cv2.imwrite(os.path.join(target_path, filename_target), target_d_map_rescaled)


def save_displacement_map(depth_map, path, filename, normalize=False, format='png'):
    """
    Save a displacement map to the specified path with format-appropriate handling.
    """
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Strip any existing extension from filename
    base_filename = os.path.splitext(filename)[0]

    # Ensure depth map is float32 for processing
    depth_map = depth_map.astype(np.float32)

    # Normalize if requested
    if normalize:
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)

    # Save in the appropriate format
    if format.lower() == 'png':
        # 16-bit PNG for better precision than 8-bit
        output_path = os.path.join(path, f"{base_filename}.png")
        depth_map_scaled = (depth_map * 65535).astype(np.uint16)
        cv2.imwrite(output_path, depth_map_scaled)

    elif format.lower() == 'tiff':
        # 32-bit TIFF can store floating point values
        output_path = os.path.join(path, f"{base_filename}.tiff")
        cv2.imwrite(output_path, depth_map)

    elif format.lower() == 'exr':
        # EXR format for full 32-bit float precision
        output_path = os.path.join(path, f"{base_filename}.exr")

        try:
            # Try using OpenCV's EXR writer
            cv2.imwrite(output_path, depth_map)
        except:
            try:
                # Prepare header
                header = OpenEXR.Header(depth_map.shape[1], depth_map.shape[0])
                half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                header['channels'] = dict([(c, half_chan) for c in "RGB"])

                # Convert data
                data = depth_map.tobytes()

                # Write EXR
                exr = OpenEXR.OutputFile(output_path, header)
                exr.writePixels({'R': data, 'G': data, 'B': data})
                exr.close()
            except ImportError:
                # Final fallback to imageio
                imageio.imwrite(output_path, depth_map)

    elif format.lower() == 'npy':
        # Native NumPy format preserves exact values
        output_path = os.path.join(path, f"{base_filename}.npy")
        np.save(output_path, depth_map)

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'png', 'tiff', 'exr', or 'npy'")

    return output_path


def validate_directories(paths):
    # Validate and create directories
    for path in paths:
        print(f"Validating directory: {path}")

        if not os.path.exists(path):
            os.makedirs(path)

        if not os.access(path, os.W_OK):
            raise Exception(f"Directory {path} is not writable.")

    print('Directories validated successfully.')


def get_image_paths(directory):
    return [
        os.path.join(directory, fname)

        for fname in sorted(os.listdir(directory))

        if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS
    ]

