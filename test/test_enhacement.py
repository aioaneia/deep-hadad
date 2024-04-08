import cv2
import utils.cv_enhancement_utils as en
import utils.cv_augmentation_utils as aug
# Load the image
image = cv2.imread('image.png')

# Sharpen the image and add it to the list
enhanced_image = en.sharp_image(image.copy())

augmented_images = aug.augment_image(enhanced_image, pipeline=aug.damaging_augmentation_pipeline)
