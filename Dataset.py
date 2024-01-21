from FunMatrix import *
from utils import *
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as T

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_path, poses, transform, K):
        self.sequence_path = sequence_path
        self.sequence_num = sequence_path.split('/')[1]
        self.poses = poses
        self.transform = transform
        self.k = K

    def __len__(self):
        return len(self.poses) - jump_frames

    def __getitem__(self, idx):
        # If one of the frames is "Bad"- skip 
        img1_path = os.path.join(self.sequence_path, f'{idx:06}.png')
        img2_path = os.path.join(self.sequence_path, f'{idx+jump_frames:06}.png')
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            return None, None, None, None

        # Create PIL images
        original_first_image = Image.open(img1_path)
        original_second_image = Image.open(img2_path)

        # Transform: Resize, center, grayscale
        first_image = self.transform(original_first_image)
        second_image = self.transform(original_second_image)

        # Adjust K according to resize and center crop transforms and compute ground-truth F matrix     
        adjusted_K = adjust_intrinsic(self.k.copy(), np.array(original_first_image.size), np.array([256, 256]), np.array([224, 224]))
        F = get_F(self.poses, idx, adjusted_K)

        # Convert to tensor and normalize F-Matrix 
        F, unnormalized_F  =  normalize_L2(normalize_L1(torch.tensor(F, dtype=torch.float32))), torch.tensor(F, dtype=torch.float32)

        return first_image, second_image, F, unnormalized_F
    
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(), # Converts to tensor and rescales [0,255] -> [0,1] 
    # TODO: Normalize images?
])

sequence_paths = [f'sequences/0{i}/image_0' for i in range(9)]
poses_paths = [f'poses/0{i}.txt' for i in range(9)]
calib_paths = [f'sequences/0{i}/calib.txt' for i in range(9)]

train_datasets, val_datasets = [], []
for i, (sequence_path, poses_path, calib_path) in enumerate(zip(sequence_paths, poses_paths, calib_paths)):
    # Get a list of all poses [R,t] in this sequence
    poses = read_poses(poses_path)

    # Read the calib.txt file to get the projection matrix to compute intrinsic K
    projection_matrix = read_calib(calib_path)
    K, _ = get_intrinsic(projection_matrix)


    # Split the dataset based on the calculated samples. Get 00 and 01 as val and the rest as train sets.
    if i < 2:
        val_datasets.append(CustomDataset(sequence_path, poses, transform, K))
    else:
        train_datasets.append(CustomDataset(sequence_path, poses, transform, K))
    

# Concatenate datasets
concat_train_dataset = ConcatDataset(train_datasets)
concat_val_dataset = ConcatDataset(val_datasets)

# Create a DataLoader
train_loader = DataLoader(concat_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(concat_val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


# bad_frames_dir = os.path.join('epipole_lines', "bad_frames")
# good_frames_dir = os.path.join('epipole_lines', "good_frames")

# os.makedirs(bad_frames_dir, exist_ok=True)
# os.makedirs(good_frames_dir, exist_ok=True)


# for i, (first_image, second_image, label, unormalized_label, idx, sequence_num) in enumerate(val_loader):
#     dst_dir = os.path.join('sequences', sequence_num[0], "BadFrames")
#     os.makedirs(dst_dir, exist_ok=True)

#     epipolar_geo = EpipolarGeometry(first_image[0], second_image[0], F=unormalized_label, idx=idx.item(), sequence_num=sequence_num[0])
#     epipolar_geo.visualize(sqResultDir='epipole_lines', img_idx=i)
#     # err += epipolar_geo.get_epipolar_err()

# for i, (first_image, second_image, label, unormalized_label, idx, sequence_num) in enumerate(train_loader):
#     dst_dir = os.path.join('sequences', sequence_num[0], "BadFrames")
#     os.makedirs(dst_dir, exist_ok=True)

#     epipolar_geo = EpipolarGeometry(first_image[0], second_image[0], F=unormalized_label, idx=idx.item(), sequence_num=sequence_num[0])
#     epipolar_geo.visualize(sqResultDir='epipole_lines', img_idx=i)
# # err /= len(train_loader) 
# # print(err)