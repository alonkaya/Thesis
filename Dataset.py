from FunMatrix import *
from utils import *
from DatasetOneSequence import CustomDataset_first_two_thirds_train, CustomDataset_first_two_out_of_three_train
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import v2
import os
from PIL import Image
import torchvision


class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequence_path, poses, valid_indices, transform, K):
        self.sequence_path = sequence_path
        self.poses = poses
        self.transform = transform
        self.k = K
        self.valid_indices = valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        idx = self.valid_indices[idx]
        
        original_first_image = torchvision.io.read_image(os.path.join(self.sequence_path, f'{idx:06}.{IMAGE_TYPE}'))
        original_second_image = torchvision.io.read_image(os.path.join(self.sequence_path, f'{idx+JUMP_FRAMES:06}.{IMAGE_TYPE}'))

        first_image = self.transform(original_first_image)
        second_image = self.transform(original_second_image)

        unnormalized_F = get_F(self.poses, idx, self.k)

        # Normalize F-Matrix
        F = norm_layer(unnormalized_F.view(-1, 9)).view(3,3)

        return first_image, second_image, F, idx, self.sequence_path

def get_valid_indices(sequence_len, sequence_path):
    valid_indices = []
    for idx in range(sequence_len - JUMP_FRAMES):
        img1_path = os.path.join(sequence_path, f'{idx:06}.{IMAGE_TYPE}')
        img2_path = os.path.join(sequence_path, f'{idx+JUMP_FRAMES:06}.{IMAGE_TYPE}')

        if os.path.exists(img1_path) and os.path.exists(img2_path):
            valid_indices.append(idx)

    return valid_indices

if AUGMENTATION:
    transform = v2.Compose([
        v2.Resize((256, 256), antialias=True),
        v2.CenterCrop(224),
        v2.Grayscale(num_output_channels=3),
        v2.ColorJitter(brightness=(0.85, 1.15), contrast=(0.85, 1.15)),
        v2.ToDtype(torch.float32, scale=True),  # Converts to torch.float32 and scales [0,255] -> [0,1]
        v2.Normalize(mean=norm_mean,  # Normalize each channel
                            std=norm_std),
    ])    
else:
    transform = v2.Compose([
        v2.Resize((256, 256), antialias=True),
        v2.CenterCrop(224),
        v2.Grayscale(num_output_channels=3),
        v2.ToDtype(torch.float32, scale=True),  # Converts to torch.float32 and scales [0,255] -> [0,1]
        v2.Normalize(mean=norm_mean,  # Normalize each channel
                            std=norm_std),
    ])     


def get_dataloaders_RealEstate(batch_size):
    RealEstate_paths = ['RealEstate10K/train_images', 'RealEstate10K/val_images']

    train_datasets, val_datasets = [], []
    for RealEstate_path in RealEstate_paths:
        for i, sequence_name in enumerate(os.listdir(RealEstate_path)):
            specs_path = os.path.join(RealEstate_path, sequence_name, f'{sequence_name}.txt')
            sequence_path = os.path.join(RealEstate_path, sequence_name, 'image_0')

            # Get a list of all poses [R,t] in this sequence
            poses = read_poses(specs_path)

            # Indices of 'good' image frames
            valid_indices = get_valid_indices(len(poses), sequence_path)
            if len(valid_indices) == 0: continue

            # Get projection matrix from calib.txt, compute intrinsic K, and adjust K according to transformations
            original_image_size = torch.tensor(Image.open(os.path.join(sequence_path, f'{valid_indices[0]:06}.{IMAGE_TYPE}')).size)
            K = get_intrinsic_REALESTATE(specs_path, original_image_size)
            
            if not FIRST_2_THRIDS_TRAIN and not FIRST_2_OF_3_TRAIN:
                custom_dataset = Dataset(sequence_path, poses, valid_indices, transform, K)
                # if len(custom_dataset) > 20:
                if RealEstate_path == 'RealEstate10K/train_images':
                    train_datasets.append(custom_dataset) 
                else:    
                    val_datasets.append(custom_dataset)
            else:
                train_dataset = CustomDataset_first_two_thirds_train(sequence_path, poses, valid_indices, transform, K, dataset_type="train") if FIRST_2_THRIDS_TRAIN else \
                                CustomDataset_first_two_out_of_three_train(sequence_path, poses, valid_indices, transform, K, dataset_type="train")
                val_dataset = CustomDataset_first_two_thirds_train(sequence_path, poses, valid_indices, transform, K, dataset_type="val") if FIRST_2_THRIDS_TRAIN else \
                                CustomDataset_first_two_out_of_three_train(sequence_path, poses, valid_indices, transform, K, dataset_type="val")
                if len(val_dataset) > 25:
                    train_datasets.append(train_dataset)
                    val_datasets.append(val_dataset)
                    
    # Concatenate datasets
    concat_train_dataset = ConcatDataset(train_datasets)
    concat_val_dataset = ConcatDataset(val_datasets)

    # Create a DataLoader
    train_loader = DataLoader(concat_train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(concat_val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader

def get_dataloaders_KITTI(batch_size):
    sequence_paths = [f'sequences/0{i}/image_0' for i in range(9)]
    poses_paths = [f'poses/0{i}.txt' for i in range(9)]
    calib_paths = [f'sequences/0{i}/calib.txt' for i in range(9)]

    train_datasets, val_datasets = [], []
    for i, (sequence_path, poses_path, calib_path) in enumerate(zip(sequence_paths, poses_paths, calib_paths)):
        if i not in train_seqeunces and i not in val_sequences: continue
        
        # Get a list of all poses [R,t] in this sequence
        poses = read_poses(poses_path)

        # Indices of 'good' image frames
        valid_indices = get_valid_indices(len(poses), sequence_path)
    
        # Get projection matrix from calib.txt, compute intrinsic K, and adjust K according to transformations
        original_image_size = torch.tensor(Image.open(os.path.join(sequence_path, f'{valid_indices[0]:06}.{IMAGE_TYPE}')).size)
        K = get_intrinsic_KITTI(calib_path, original_image_size)

        # Split the dataset based on the calculated samples. Get 00 and 01 as val and the rest as train sets.
        custom_dataset = Dataset(sequence_path, poses, valid_indices, transform, K)
        if i in train_seqeunces:
            train_datasets.append(custom_dataset)        
        elif i in val_sequences:
            val_datasets.append(custom_dataset)

    # Concatenate datasets
    concat_train_dataset = ConcatDataset(train_datasets)
    concat_val_dataset = ConcatDataset(val_datasets)

    # Create a DataLoader
    train_loader = DataLoader(concat_train_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(concat_val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader


def get_data_loaders(batch_size):
    if USE_REALESTATE:
        return get_dataloaders_RealEstate(batch_size)
    else: # KITTI
        return get_dataloaders_KITTI(batch_size)



