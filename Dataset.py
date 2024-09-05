from torch.utils.data import DataLoader
import torch
from datasets import load_dataset
import torchvision.transforms.functional as F
import random
import numpy as np
from params import *
import torchvision.transforms as v2


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, angle_range=None, shift_range=None):
        self.dataset = dataset
        self.transform = transform
        self.angle_range = angle_range
        self.shift_range = shift_range

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        original_image = self.dataset[idx]['image']

        # Transform: Resize, center, grayscale
        original_image = self.transform(original_image)

        # Generate random affine params
        angle = random.uniform(-self.angle_range, self.angle_range)
        shift_x = random.uniform(-self.shift_x_range, self.shift_x_range)
        shift_y = random.uniform(-self.shift_y_range, self.shift_y_range)

        translated_image = F.affine(original_image, angle=angle, translate=(shift_x, shift_y), scale=1, shear=0)

        # Convert both images to tensor and rescale [0,255] -> [0,1]
        translated_image, original_image  = F.to_tensor(translated_image), F.to_tensor(original_image)

        # Rescale params -> [0,1]
        angle = 0 if self.angle_range == 0 else abs(angle) / self.angle_range
        shift_x = 0 if self.shift_x_range == 0 else abs(shift_x) / self.shift_x_range
        shift_y = 0 if self.shift_y_range == 0 else abs(shift_y) / self.shift_y_range

        # Convert params to tensor
        if angle == 0:
            params = torch.tensor([shift_x, shift_y], dtype=torch.float32)
        elif shift_x == 0 or shift_y == 0:
            params = torch.tensor([angle], dtype=torch.float32)
        else:
            params = torch.tensor([angle, shift_x, shift_y], dtype=torch.float32)

        return original_image, translated_image, params


def get_dataloaders(batch_size, num_of_training_images, num_of_val_images):
    transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.Grayscale(num_output_channels=3),
        v2.ToTensor(), # Converts to torch.float32 and scales [0,255] -> [0,1]
        v2.Normalize(mean=norm_mean.to(device), std=norm_std.to(device))  # Normalize each channel
    ])

    # Load and display the image
    dataset = load_dataset("frgfm/imagenette", "320px")
    train_dataset = dataset["train"].select(np.arange(num_of_training_images))
    val_dataset = dataset["validation"].select(np.arange(num_of_val_images))

    # Create an instance dataset
    custom_train_dataset = CustomDataset(train_dataset, transform=transform, angle_range=angle_range, shift_range=shift_range)
    custom_val_dataset = CustomDataset(val_dataset, transform=transform,angle_range=angle_range, shift_range=shift_range)

    # Create a DataLoader
    train_loader = DataLoader(custom_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(custom_val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=32, num_of_training_images=100, num_of_val_images=10)
    