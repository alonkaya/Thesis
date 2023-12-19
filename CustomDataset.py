from params import *
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from datasets import load_dataset, Dataset, Image
import torchvision.transforms.functional as F
import random
import numpy as np


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, angle_range=None, shift_x_range=None, shift_y_range=None):
        self.dataset = dataset
        self.transform = transform
        self.angle_range = angle_range
        self.shift_x_range=shift_x_range
        self.shift_y_range=shift_y_range

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        original_image = self.dataset[idx]['image']

        # Transform: Resize, center, grayscale
        original_image = self.transform(original_image)

        # Generate random affine params
        angle = 0 if self.angle_range == 0 else random.uniform(-self.angle_range, self.angle_range)
        shift_x = 0 if self.shift_x_range == 0 else random.uniform(-self.shift_x_range, self.shift_x_range)
        shift_y = 0 if self.shift_y_range == 0 else random.uniform(-self.shift_y_range, self.shift_y_range)

        translated_image = F.affine(original_image, angle=angle, translate=(shift_x, shift_y), scale=1, shear=0)

        # Convert both images to tensor and rescale [0,255] -> [0,1]
        translated_image, original_image  = F.to_tensor(translated_image), F.to_tensor(original_image)

        # Rescale params -> [0,1]
        angle = 0 if self.angle_range == 0 else abs(angle) / self.angle_range
        shift_x = 0 if self.shift_x_range == 0 else abs(shift_x) / self.shift_x_range
        shift_y = 0 if self.shift_y_range == 0 else abs(shift_y) / self.shift_y_range
        
        # Convert params to tensor
        params = torch.tensor([shift_x, shift_y], dtype=torch.float32)

        # TODO: normalize images
        return original_image, translated_image, params

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
])
# Load the dataset
# image_paths = ["/content/drive/MyDrive/Thesis/image.jpg", "/content/drive/MyDrive/Thesis/image2.jpg", "/content/drive/MyDrive/Thesis/image3.jpg", "/content/drive/MyDrive/Thesis/image4.jpg", "/content/drive/MyDrive/Thesis/image5.jpg"]
# dataset = Dataset.from_dict({"image": image_paths}).cast_column("image", Image())

# Load and display the image
dataset = load_dataset("frgfm/imagenette", 'full_size')
train_dataset = dataset["train"].select(np.arange(num_of_training_images))
val_dataset = dataset["validation"].select(np.arange(num_of_val_images))

# Create an instance dataset
custom_train_dataset = CustomDataset(train_dataset, transform=transform, angle_range=angle_range, shift_x_range=shift_x_range, shift_y_range=shift_y_range)
custom_val_dataset = CustomDataset(val_dataset, transform=transform,angle_range=angle_range, shift_x_range=shift_x_range, shift_y_range=shift_y_range)

# Create a DataLoader
train_loader = DataLoader(custom_train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(custom_val_dataset, batch_size=32, shuffle=False, pin_memory=True)