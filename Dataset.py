# from params import *
# from FunMatrix import *
# from utils import generate_pose_and_frame_numbers, normalize_L1, normalize_L2
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms, datasets
# import torch
# import os
# from PIL import Image
# import matplotlib.pyplot as plt
# import torchvision.transforms.functional as T


# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, sequence_path, poses, frame_numbers, transform, K):
#         self.sequence_path = sequence_path
#         self.poses = poses
#         self.frame_numbers = frame_numbers
#         self.transform = transform
#         self.k = K

#     def __len__(self):
#         return len(self.frame_numbers) - jump_frames

#     def __getitem__(self, idx):
#         # Create PIL images
#         first_image = Image.open(os.path.join(self.sequence_path, f'{self.frame_numbers[idx]:06}.png'))
#         second_image = Image.open(os.path.join(self.sequence_path, f'{self.frame_numbers[idx+jump_frames]:06}.png'))

#         # Transform: Resize, center, grayscale
#         first_image = self.transform(first_image)
#         second_image = self.transform(second_image)

#         # Compute relative rotation and translation matrices
#         R_relative, t_relative = compute_relative_transformations(self.poses[idx], self.poses[idx+jump_frames])

#         # # Compute the essential matrix E
#         E = compute_essential(R_relative, t_relative)

#         # Compute the fundamental matrix F
#         F = compute_fundamental(E, self.k, self.k)

#         # Convert to tensor and rescale [0,255] -> [0,1]
#         first_image, second_image, F, unnormalized_F  = T.to_tensor(first_image), T.to_tensor(second_image), normalize_L2(normalize_L1(torch.tensor(F, dtype=torch.float32))), torch.tensor(F, dtype=torch.float32)
        
#         # TODO: Normalize images?
#         return first_image, second_image, F, unnormalized_F

# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.Grayscale(num_output_channels=3),
# ])

# sequence_path = 'sequences/02/image_0'
# poses_path = 'poses/02.txt'
# calib_path = 'sequences/02/calib.txt'

# poses, frame_numbers = generate_pose_and_frame_numbers(poses_path)

# # Read the calib.txt file to get the projection matricx and compute intrinsic K
# projection_matrix = process_calib(calib_path)
# K, _ = get_internal_param_matrix(projection_matrix)

# # dataset = CustomDataset(sequence_path, poses, frame_numbers, transform, K)

# # train_len = int(0.8 * len(dataset))
# # val_len = len(dataset) - train_len

# # Split train validation sets
# # train_dataset, val_dataset =  random_split(dataset, [train_len, val_len])

# # Calculate the number of samples for training and validation
# train_samples = int(train_ratio * (len(frame_numbers)-1))

# # Split the dataset based on the calculated samples. Get first train_samples of data for training and the rest for validation
# train_dataset = CustomDataset(sequence_path, poses[:train_samples], frame_numbers[:train_samples], transform, K)
# val_dataset = CustomDataset(sequence_path, poses[train_samples:], frame_numbers[train_samples:], transform, K)

# # Create a DataLoader
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)

# # Visualize an image:
# # train_iter = iter(train_loader)
# # first_image, second_image, label = next(train_iter)
# # plt.imshow(first_image[0].permute(1, 2, 0).cpu().numpy())
# # plt.axis('off')
# # plt.show()

from FunMatrix import *
from utils import *
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
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
        self.valid_indices = self.get_valid_indices()

    def __len__(self):
        return len(self.valid_indices) - jump_frames

    def get_valid_indices(self):
        valid_indices = []
        for idx in range(len(self.poses) - jump_frames):
            img1_path = os.path.join(self.sequence_path, f'{idx:06}.png')
            img2_path = os.path.join(self.sequence_path, f'{idx+jump_frames:06}.png')
            if os.path.exists(img1_path) and os.path.exists(img2_path):
                valid_indices.append(idx)
        return valid_indices

    def __getitem__(self, idx):
        idx = self.valid_indices[idx]
        original_first_image = Image.open(os.path.join(self.sequence_path, f'{idx:06}.png'))
        original_second_image = Image.open(os.path.join(self.sequence_path, f'{idx+jump_frames:06}.png'))
        # if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        #     return None  # Return None if images don't exist

        # Transform: Resize, center, grayscale
        first_image = self.transform(original_first_image).to(device)
        second_image = self.transform(original_second_image).to(device)

        # Adjust K according to resize and center crop transforms and compute ground-truth F matrix
        adjusted_K = adjust_intrinsic(self.k.clone(), torch.tensor(original_first_image.size).to(device), torch.tensor([256, 256]).to(device), torch.tensor([224, 224]).to(device))
        
        unnormalized_F = get_F(self.poses, idx, adjusted_K)

        # Normalize F-Matrix
        F = normalize_L2(normalize_L1(unnormalized_F))

        return first_image, second_image, F, unnormalized_F



def get_data_loaders():

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),  # Converts to tensor and rescales [0,255] -> [0,1]
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
        # K = get_intrinsic(calib_path)
        projection_matrix = process_calib(calib_path)
        K, _ = get_internal_param_matrix(projection_matrix)

        # Split the dataset based on the calculated samples. Get 00 and 01 as val and the rest as train sets.
        if i in train_seqeunces:
            train_datasets.append(CustomDataset(
                sequence_path, poses, transform, K))        
        elif i in val_sequences:
            val_datasets.append(CustomDataset(
                sequence_path, poses, transform, K))


    # Concatenate datasets
    concat_train_dataset = ConcatDataset(train_datasets)
    concat_val_dataset = ConcatDataset(val_datasets)

    # Create a DataLoader
    train_loader = DataLoader(concat_train_dataset,
                              batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(
        concat_val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)    

    return train_loader, val_loader


def visualize_bad_images():
    bad_frames_dir = os.path.join('epipole_lines', "bad_frames")
    good_frames_dir = os.path.join('epipole_lines', "good_frames")

    os.makedirs(bad_frames_dir, exist_ok=True)
    os.makedirs(good_frames_dir, exist_ok=True)


# def move_bad_images():
#     # change dataset returns 6 params instead of 4. comment unnecessary lines in visualize
#     train_loader, val_loader = get_data_loaders()

#     for i, (first_image, second_image, label, unormalized_label, idx, sequence_num) in enumerate(val_loader):
#         if first_image[0].shape == ():
#             continue
#         dst_dir = os.path.join('sequences', sequence_num[0], "BadFrames")
#         os.makedirs(dst_dir, exist_ok=True)

#         epipolar_geo = EpipolarGeometry(
#             first_image[0], second_image[0], F=unormalized_label, idx=idx.item(), sequence_num=sequence_num[0])
#         epipolar_geo.visualize(sqResultDir='epipole_lines', img_idx=i)

#     for i, (first_image, second_image, label, unormalized_label, idx, sequence_num) in enumerate(train_loader):
#         if first_image[0].shape == ():
#             continue
#         dst_dir = os.path.join('sequences', sequence_num[0], "BadFrames")
#         os.makedirs(dst_dir, exist_ok=True)

#         epipolar_geo = EpipolarGeometry(
#             first_image[0], second_image[0], F=unormalized_label, idx=idx.item(), sequence_num=sequence_num[0])
#         epipolar_geo.visualize(sqResultDir='epipole_lines', img_idx=i)

# def test_ground_truth_epipolar_err():
#     """computes average epipolar error for both normalized ground truth and unnormalized ground truth """

#     train_loader, val_loader = get_data_loaders()
    
#     avg_ep_err_unnormalized, avg_ep_err = 0, 0
#     for first_image, second_image, label, unormalized_label in train_loader:
#         ep_err_unnormalized, ep_err = 0, 0
#         for img_1, img_2, F, unormalized_F in zip(first_image, second_image, label, unormalized_label):
#             ep_err_unnormalized += EpipolarGeometry(img_1, img_2, unormalized_F).get_epipolar_err()
#             ep_err += EpipolarGeometry(img_1, img_2, F).get_epipolar_err()

#         ep_err_unnormalized, ep_err = ep_err_unnormalized/len(first_image), ep_err/len(first_image)
#         avg_ep_err_unnormalized, avg_ep_err = avg_ep_err_unnormalized + ep_err_unnormalized, avg_ep_err + ep_err

#     avg_ep_err_unnormalized, avg_ep_err = avg_ep_err_unnormalized/len(train_loader), avg_ep_err/len(train_loader)
#     return avg_ep_err_unnormalized, avg_ep_err
