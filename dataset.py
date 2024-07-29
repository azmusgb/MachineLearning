from torch.utils.data import Dataset
import numpy as np
import torch

class AugmentedDataset(Dataset):
    def __init__(self, original_image, transform=None, num_augments=100):
        self.original_image = original_image
        self.transform = transform
        self.num_augments = num_augments

    def __len__(self):
        return self.num_augments

    def __getitem__(self, idx):
        image = self.original_image.copy()
        if self.transform:
            image = self.transform(images=[image])[0]
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        label = idx % 10
        return torch.tensor(image), torch.tensor(label)
