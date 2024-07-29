import cv2
import numpy as np
from imgaug import augmenters as iaa
from typing import List, Tuple

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to load image at {image_path}")
    return image

def preprocess_image(image, blur_ksize):
    img = cv2.equalizeHist(image)
    img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img

def segment_text_regions(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 100]
    return regions

def augment_image(image: np.ndarray, num_augments: int = 5) -> List[np.ndarray]:
    """Generate augmented versions of the input image."""
    augmentation_seq = iaa.Sequential([
        iaa.Affine(rotate=(-10, 10)),
        iaa.ScaleX((0.8, 1.2)),
        iaa.ScaleY((0.8, 1.2)),
        iaa.AdditiveGaussianNoise(scale=(10, 30)),
        iaa.GaussianBlur(sigma=(0.0, 3.0)),
        iaa.ShearX((-10, 10)),
        iaa.ShearY((-10, 10)),
        iaa.contrast.LinearContrast((0.75, 1.5)),
    ])
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    images_aug = augmentation_seq(images=[image] * num_augments)
    return images_aug
