from FunMatrix import *
from utils import *
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as T


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_path, poses, valid_indices, transform, K):
        self.sequence_path = sequence_path
        self.sequence_num = sequence_path.split('/')[1]
        self.poses = poses
        self.transform = transform
        self.k = K
        self.valid_indices = valid_indices

    def __len__(self):
        return len(self.valid_indices) - JUMP_FRAMES

    def get_valid_indices(self):
        valid_indices = []
        for idx in range(len(self.poses) - JUMP_FRAMES):
            img1_path = os.path.join(self.sequence_path, f'{idx:06}.{IMAGE_TYPE}')
            img2_path = os.path.join(self.sequence_path, f'{idx+JUMP_FRAMES:06}.{IMAGE_TYPE}')
            if os.path.exists(img1_path) and os.path.exists(img2_path):
                valid_indices.append(idx)
        return valid_indices

    def __getitem__(self, idx):
        idx = self.valid_indices[idx]
        
        original_first_image = Image.open(os.path.join(self.sequence_path, f'{idx:06}.{IMAGE_TYPE}'))
        original_second_image = Image.open(os.path.join(self.sequence_path, f'{idx+JUMP_FRAMES:06}.{IMAGE_TYPE}'))

        # Transform: Resize, center, grayscale
        first_image = self.transform(original_first_image).to(device)
        second_image = self.transform(original_second_image).to(device)

        unnormalized_F = get_F(self.poses, idx, self.k)

        # Normalize F-Matrix
        F = norm_layer(unnormalized_F.view(-1, 9)).view(3,3)

        return first_image, second_image, F, unnormalized_F

def get_valid_indices(sequence_len, sequence_path):
    valid_indices = []
    for idx in range(sequence_len - JUMP_FRAMES):
        img1_path = os.path.join(sequence_path, f'{idx:06}.{IMAGE_TYPE}')
        img2_path = os.path.join(sequence_path, f'{idx+JUMP_FRAMES:06}.{IMAGE_TYPE}')
        if os.path.exists(img1_path): print(img1_path)
        if os.path.exists(img1_path) and os.path.exists(img2_path):
            valid_indices.append(idx)
    return valid_indices

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),  # Converts to tensor and rescales [0,255] -> [0,1]
    # TODO: Normalize images?
])    

def get_dataloaders_KITTI(batch_size):
    sequence_paths = [f'sequences/0{i}/image_0' for i in range(9)]
    poses_paths = [f'poses/0{i}.txt' for i in range(9)]
    calib_paths = [f'sequences/0{i}/calib.txt' for i in range(9)]

    train_datasets, val_datasets = [], []
    for i, (sequence_path, poses_path, calib_path) in enumerate(zip(sequence_paths, poses_paths, calib_paths)):
        if i not in train_seqeunces and i not in val_sequences: continue
        
        # Get a list of all poses [R,t] in this sequence
        poses = read_poses(poses_path).to(device)

        # Indices of 'good' image frames
        valid_indices = get_valid_indices(len(poses), sequence_path)
    
        # Get projection matrix from calib.txt, compute intrinsic K, and adjust K according to transformations
        original_image_size = torch.tensor(Image.open(os.path.join(sequence_path, f'{valid_indices[0]:06}.{IMAGE_TYPE}')).size).to(device)
        K = get_intrinsic_KITTI(calib_path, original_image_size)

        # Split the dataset based on the calculated samples. Get 00 and 01 as val and the rest as train sets.
        if i in train_seqeunces:
            train_datasets.append(CustomDataset(sequence_path, poses, valid_indices, transform, K))        
        elif i in val_sequences:
            val_datasets.append(CustomDataset(sequence_path, poses, valid_indices, transform, K))

    # Concatenate datasets
    concat_train_dataset = ConcatDataset(train_datasets)
    concat_val_dataset = ConcatDataset(val_datasets)

    # Create a DataLoader
    train_loader = DataLoader(concat_train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(concat_val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)    

    return train_loader, val_loader

def get_dataloaders_RealEstate(batch_size):
    RealEstate_paths = ['RealEstate10K/train_images', 'RealEstate10K/validation_images']

    train_datasets, val_datasets = [], []
    for RealEstate_path in RealEstate_paths:
        for sequence_name in os.listdir(RealEstate_path):
            specs_path = os.path.join(RealEstate_path, sequence_name, f'{sequence_name}.txt')
            sequence_path = os.path.join(RealEstate_path, sequence_name, 'image_0')

            # Get a list of all poses [R,t] in this sequence
            poses = read_poses(specs_path).to(device)

            # Indices of 'good' image frames
            valid_indices = get_valid_indices(len(poses), sequence_path)
        
            # Get projection matrix from calib.txt, compute intrinsic K, and adjust K according to transformations
            original_image_size = torch.tensor(Image.open(os.path.join(sequence_path, f'{valid_indices[0]:06}.{IMAGE_TYPE}')).size).to(device)
            K = get_intrinsic_REALESTATE(specs_path, original_image_size)

            if RealEstate_path == 'RealEstate10K\\train_images':
                train_datasets.append(CustomDataset(sequence_path, poses, valid_indices, transform, K))     
            else:   
                val_datasets.append(CustomDataset(sequence_path, poses, valid_indices, transform, K))

    # Concatenate datasets
    concat_train_dataset = ConcatDataset(train_datasets)
    concat_val_dataset = ConcatDataset(val_datasets)

    # Create a DataLoader
    train_loader = DataLoader(concat_train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(concat_val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)    

    return train_loader, val_loader

def get_data_loaders(batch_size):
    if USE_REALESTATE:
        return get_dataloaders_RealEstate(batch_size)
    else: # KITTI
        return get_dataloaders_KITTI(batch_size)


def move_bad_images():
    # change dataset returns 6 params instead of 4. comment unnecessary lines in visualize
    train_loader, val_loader = get_data_loaders(batch_size=1)

    for i, (first_image, second_image, label, unormalized_label, idx, sequence_num) in enumerate(val_loader):
        if first_image[0].shape == ():
            continue
        dst_dir = os.path.join('sequences', sequence_num[0], "BadFrames")
        os.makedirs(dst_dir, exist_ok=True)

        epipolar_geo = EpipolarGeometry(
            first_image[0], second_image[0], F=unormalized_label, idx=idx.item(), sequence_num=sequence_num[0])
        epipolar_geo.visualize(sqResultDir='epipole_lines', img_idx=i)

    for i, (first_image, second_image, label, unormalized_label, idx, sequence_num) in enumerate(train_loader):
        if first_image[0].shape == ():
            continue
        dst_dir = os.path.join('sequences', sequence_num[0], "BadFrames")
        os.makedirs(dst_dir, exist_ok=True)

        epipolar_geo = EpipolarGeometry(
            first_image[0], second_image[0], F=unormalized_label, idx=idx.item(), sequence_num=sequence_num[0])
        epipolar_geo.visualize(sqResultDir='epipole_lines', img_idx=i)

def test_ground_truth_epipolar_err():
    """computes average epipolar error for both normalized ground truth and unnormalized ground truth """

    train_loader, val_loader = get_data_loaders(batch_size=1)
    
    avg_ep_err_unnormalized, avg_ep_err = 0, 0
    for first_image, second_image, label, unormalized_label in val_loader:
        ep_err_unnormalized, ep_err = 0, 0
        for img_1, img_2, F, unormalized_F in zip(first_image, second_image, label, unormalized_label):
            ep_err_unnormalized += EpipolarGeometry(img_1, img_2, unormalized_F).get_epipolar_err()
            ep_err += EpipolarGeometry(img_1, img_2, F).get_epipolar_err()

        ep_err_unnormalized, ep_err = ep_err_unnormalized/len(first_image), ep_err/len(first_image)
        avg_ep_err_unnormalized, avg_ep_err = avg_ep_err_unnormalized + ep_err_unnormalized, avg_ep_err + ep_err

    avg_ep_err_unnormalized, avg_ep_err = avg_ep_err_unnormalized/len(val_loader), avg_ep_err/len(val_loader)
    return avg_ep_err_unnormalized, avg_ep_err

# if __name__ == "__main__":
#     print(test_ground_truth_epipolar_err())
