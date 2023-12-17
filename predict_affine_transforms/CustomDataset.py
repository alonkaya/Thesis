from params import *
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from datasets import load_dataset, Dataset, Image
import torchvision.transforms.functional as F
import random
import numpy as np


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, angle_range=180):
        self.dataset = dataset
        self.transform = transform
        self.angle_range = angle_range

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']

        #Generate random angle
        angle = random.uniform(-self.angle_range, self.angle_range)

        # Transform: Resize, center, grayscale
        image = self.transform(image)

        # Rotate image
        rotated_image = F.rotate(image, angle)

        # Convert both images to tensor and rescale [0,255] -> [0,1]
        rotated_image, image  = F.to_tensor(rotated_image), F.to_tensor(image)

        # Convert angle to tensor and rescale [0,255] -> [0,1]
        rescaled_angle = abs(angle) / self.angle_range
        rescaled_angle = torch.tensor(rescaled_angle, dtype=torch.float32).view(1,)

        #TODO: normalize images
        return image, rotated_image, rescaled_angle


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
custom_train_dataset = CustomDataset(train_dataset, transform=transform, angle_range=angle_range)
custom_val_dataset = CustomDataset(val_dataset, transform=transform,angle_range=angle_range)

# Create a DataLoader
train_loader = DataLoader(custom_train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(custom_val_dataset, batch_size=32, shuffle=False, pin_memory=True)
