import albumentations as A

shared_augmentation_pipeline = A.Compose([
    A.Rotate(limit=10, p=0.8),
    A.RandomScale(scale_limit=0.1, p=0.5),
    A.RandomCrop(height=384, width=384),
    A.Perspective(scale=(0.05, 0.08), p=0.5)
])


def augment_image(image, pipeline):
    return pipeline(image=image)['image']


def augment_preserved_glyph_image(image):
    return augment_image(image, pipeline=shared_augmentation_pipeline)

