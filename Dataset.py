from params import *
from FunMatrix import *
from utils import generate_pose_and_frame_numbers, normalize_L1, normalize_L2
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as T


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_path, poses, frame_numbers, transform, K):
        self.sequence_path = sequence_path
        self.poses = poses
        self.frame_numbers = frame_numbers
        self.transform = transform
        self.k = K

    def __len__(self):
        return len(self.frame_numbers) - jump_frames

    def __getitem__(self, idx):
        # Create PIL images
        first_image = Image.open(os.path.join(self.sequence_path, f'{self.frame_numbers[idx]:06}.png'))
        second_image = Image.open(os.path.join(self.sequence_path, f'{self.frame_numbers[idx+jump_frames]:06}.png'))

        # Transform: Resize, center, grayscale
        # first_image = self.transform(first_image)
        # second_image = self.transform(second_image)

        # Compute relative rotation and translation matrices
        R_relative, t_relative = compute_relative_transformations(self.poses[idx], self.poses[idx+jump_frames])

        # # Compute the essential matrix E
        E = compute_essential(R_relative, t_relative)

        # Compute the fundamental matrix F
        F = compute_fundamental(E, self.k, self.k)

        # Convert to tensor and rescale [0,255] -> [0,1]
        first_image, second_image, F  = T.to_tensor(first_image), T.to_tensor(second_image), F
        
        # TODO: Normalize images?
        return first_image, second_image, F

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
])

sequence_path = 'sequences/02/image_0'
poses_path = 'poses/02.txt'
calib_path = 'sequences/02/calib.txt'

poses, frame_numbers = generate_pose_and_frame_numbers(poses_path)

# Read the calib.txt file to get the projection matricx and compute intrinsic K
projection_matrix = process_calib(calib_path)
K, _ = get_internal_param_matrix(projection_matrix)

# dataset = CustomDataset(sequence_path, poses, frame_numbers, transform, K)

# train_len = int(0.8 * len(dataset))
# val_len = len(dataset) - train_len

# Split train validation sets
# train_dataset, val_dataset =  random_split(dataset, [train_len, val_len])

# Calculate the number of samples for training and validation
train_samples = int(train_ratio * (len(frame_numbers)-1))

# Split the dataset based on the calculated samples. Get first train_samples of data for training and the rest for validation
train_dataset = CustomDataset(sequence_path, poses[:train_samples], frame_numbers[:train_samples], transform, K)
val_dataset = CustomDataset(sequence_path, poses[train_samples:], frame_numbers[train_samples:], transform, K)

# Create a DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)

# Visualize an image:
# train_iter = iter(train_loader)
# first_image, second_image, label = next(train_iter)
# plt.imshow(first_image[0].permute(1, 2, 0).cpu().numpy())
# plt.axis('off')
# plt.show()