from torch.utils.data import Dataset
import numpy as np
import torch


class AugmentedDataset(Dataset):
    def __init__(
        self, original_image: np.ndarray, transform=None, num_augments: int = 100
    ):
        """
        Initialize the augmented dataset.

        Args:
            original_image: The original image to be augmented.
            transform: An optional transformation to apply to the image.
            num_augments: The number of augmented images to generate. Defaults to 100.
        """
        self.original_image = original_image
        self.transform = transform
        self.num_augments = num_augments

    def __len__(self) -> int:
        return self.num_augments

    def __getitem__(self, idx: int) -> tuple:
        image = self.original_image.copy()
        if self.transform:
            image = self.transform(images=[image])[0]
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        label = idx % 10
        return torch.tensor(image), torch.tensor(label)
