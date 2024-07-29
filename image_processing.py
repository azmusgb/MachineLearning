import cv2
import numpy as np
from imgaug import augmenters as iaa
from typing import List, Tuple

def load_image(image_path: str) -> np.ndarray:
    """Load an image from the specified path."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to load image at {image_path}. Check the file path and format.")
    return image

def preprocess_image(image: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
    """Preprocess the image by equalizing the histogram, blurring, and thresholding."""
    img = cv2.equalizeHist(image)
    img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img

def segment_text_regions(img: np.ndarray, min_contour_area: int = 100) -> List[Tuple[int, int, int, int]]:
    """Segment the text regions from the image using contour detection."""
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_contour_area]
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
    if image.ndim == 2:  # Ensure the image has a channel dimension for imgaug
        image = np.expand_dims(image, axis=-1)
    images_aug = augmentation_seq(images=[image] * num_augments)
    return images_aug

# Example Usage:
if __name__ == "__main__":
    image_path = 'path/to/image.jpg'
    output_dir = Path('output')

    # Load and preprocess the image
    image = load_image(image_path)
    processed_image = preprocess_image(image)

    # Segment text regions
    regions = segment_text_regions(processed_image)
    print("Detected regions:", regions)

    # Generate augmented images
    augmented_images = augment_image(processed_image)
    print("Generated augmented images:", len(augmented_images))
