import os
from torch.utils.data import DataLoader
import torch
from datasets import load_dataset
import torchvision.transforms.functional as F
import random
import numpy as np
from params import *
import torchvision.transforms as v2

from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt


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
        original_image = self.transform(original_image).to(device)

        # Generate random affine params
        angle, shift_x, shift_y = random.uniform(-self.angle_range, self.angle_range), random.uniform(-self.shift_range, self.shift_range), random.uniform(-self.shift_range, self.shift_range)

        translated_image = F.affine(original_image, angle=angle, translate=(shift_x, shift_y), scale=1, shear=0).to(device)

        # Rescale params -> [0,1]
        angle = torch.tensor(abs(angle) / self.angle_range, dtype=torch.float32).to(device)
        shift_x = torch.tensor(abs(shift_x) / self.shift_range, dtype=torch.float32).to(device)
        shift_y = torch.tensor(abs(shift_y) / self.shift_range, dtype=torch.float32).to(device)

        return original_image, translated_image, angle, shift_x, shift_y


def get_dataloaders(batch_size=BATCH_SIZE, train_length=train_length, val_length=val_length, test_length=test_length):
    transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.Grayscale(num_output_channels=3),
        v2.ToTensor(), 
        v2.Normalize(mean, std)
    ])

    # Load and display the image
    dataset = load_dataset("frgfm/imagenette", "320px")

    # Further split the test data into validation and test sets (e.g., 50% validation, 50% test)
    train_data = dataset["train"].select(range(train_length))
    val_data = dataset['validation'].select(range(val_length))
    test_data = dataset['validation'].select(range(len(dataset['validation']) - test_length, len(dataset['validation'])))

    # Create an instance dataset
    custom_train_dataset = CustomDataset(train_data, transform=transform, angle_range=angle_range, shift_range=shift_range)
    custom_val_dataset = CustomDataset(val_data, transform=transform,angle_range=angle_range, shift_range=shift_range)
    custom_test_dataset = CustomDataset(test_data, transform=transform,angle_range=angle_range, shift_range=shift_range)

    # # Create a DataLoader
    train_loader = DataLoader(custom_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(custom_val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
    test_loader = DataLoader(custom_test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

    return train_loader, val_loader, test_loader

# Function to display original and rotated images using PIL
def show_images(original_image, rotated_image, mean=mean, std=std):
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    
    original_pil_image = to_pil_image(original_image * std + mean)
    rotated_pil_image = to_pil_image(rotated_image * std + mean)

    # Display the original and rotated images using PIL
    original_pil_image.show(title="Original Image")
    rotated_pil_image.show(title="Rotated Image")

    # Optionally, save the images if needed
    original_pil_image.save("original_image.png")
    rotated_pil_image.save("rotated_image.png")

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32, train_length=train_length, val_length=val_length, test_length=test_length)
    
    # Get a batch of images
    it = iter(train_loader)
    original_image, rotated_image, _, _, _ = next(it)

    # Visualize the first image in the batch
    show_images(original_image[0], rotated_image[0])  # Display the first image from the batch
