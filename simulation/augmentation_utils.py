import albumentations as A
import cv2

shared_augmentation_pipeline = A.Compose([
    A.Rotate(limit=20, p=0.5),  # Moderate rotation
    A.RandomScale(scale_limit=0.1, p=0.5),  # Slight scaling to simulate distance variations
    # A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=0),
    # A.RandomCrop(height=512, width=512, always_apply=True),
    A.Perspective(scale=(0.05, 0.1), p=0.5),  # Moderate perspective transformations
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),  # Subtle elastic transformations
])

damaging_augmentation_pipeline = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.7),
    A.GaussNoise(var_limit=(20, 60), p=0.6),
    A.RandomGamma(gamma_limit=(80, 120), p=0.6),
    A.ElasticTransform(alpha=2, sigma=50, alpha_affine=40, p=0.5),
    A.CoarseDropout(max_holes=10, max_height=20, max_width=20, min_holes=3, fill_value=0, p=0.6),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.Rotate(limit=15, p=0.5),
    # Replace 'desired_height' and 'desired_width' with actual values
    A.RandomCrop(height=512, width=512, p=0.5),
])

preserved_glyph_augmentation_pipeline = A.Compose([
    A.Rotate(limit=35, p=0.5),  # Rotate the image by 15 degrees
    A.RandomCrop(height=512, width=512, p=0.5),
])


def augment_image(image, pipeline):
    return pipeline(image=image)['image']


def augment_preserved_glyph_image(image):
    return augment_image(image, pipeline=shared_augmentation_pipeline)
