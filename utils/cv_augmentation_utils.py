
import albumentations as A


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


def augment_image(image, pipeline=damaging_augmentation_pipeline):
    return pipeline(image=image)['image']
