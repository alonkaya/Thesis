import random
from FunMatrix import *
from utils import *
from DatasetOneSequence import CustomDataset_first_two_thirds_train, CustomDataset_first_two_out_of_three_train
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import v2
import os
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF


class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequence_path, poses, valid_indices, transform, k, val, seq_name, jump_frames=JUMP_FRAMES):
        self.sequence_path = sequence_path
        self.poses = poses
        self.transform = transform
        self.k = k
        self.valid_indices = valid_indices
        self.val = val
        self.seq_name = seq_name
        self.jump_frames = jump_frames

    def __len__(self):
        return (len(self.valid_indices) - VAL_LENGTH if not self.val else VAL_LENGTH) - self.jump_frames

    def __getitem__(self, idx):
        idx = self.valid_indices[idx] + VAL_LENGTH if not self.val else self.valid_indices[idx]
        
        img1 = torchvision.io.read_image(os.path.join(self.sequence_path, f'{idx:06}.{IMAGE_TYPE}'))
        img2 = torchvision.io.read_image(os.path.join(self.sequence_path, f'{idx+self.jump_frames:06}.{IMAGE_TYPE}'))
        
        k=self.k.clone()
        if RANDOM_CROP:
            img1, img2 = TF.resize(img1, (256, 256), antialias=True), TF.resize(img2, (256, 256), antialias=True)
            top_crop, left_crop = random.randint(0, 32), random.randint(0, 32)
            img1, img2 = TF.crop(img1, top_crop, left_crop, 224, 224), TF.crop(img2, top_crop, left_crop, 224, 224)
            k = adjust_k_crop(k, top_crop, left_crop)

        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        unnormalized_F = get_F(self.poses, idx, k, k, self.jump_frames)
        
        # Normalize F-Matrix
        F = norm_layer(unnormalized_F.view(-1, 9)).view(3,3)

        epi = EpipolarGeometry(img1, img2, F=F)
        pts1, pts2 = epi.pts1, epi.pts2
        print(pts1.shape)
        return img1, img2, F, pts1, pts2, self.seq_name

def get_valid_indices(sequence_len, sequence_path, jump_frames=JUMP_FRAMES):
    valid_indices = []
    for idx in range(sequence_len - jump_frames):
        img1_path = os.path.join(sequence_path, f'{idx:06}.{IMAGE_TYPE}')
        img2_path = os.path.join(sequence_path, f'{idx+jump_frames:06}.{IMAGE_TYPE}')

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
elif RANDOM_CROP:
    transform = v2.Compose([
        v2.Grayscale(num_output_channels=3),
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


def get_dataloaders_RealEstate(batch_size=BATCH_SIZE):
    RealEstate_paths = ['RealEstate10K/train_images', 'RealEstate10K/val_images']
    train_datasets, val_datasets = [], []
    for jump_frames in [JUMP_FRAMES]:
        for RealEstate_path in RealEstate_paths:
            for i, sequence_name in enumerate(os.listdir(RealEstate_path)):
                specs_path = os.path.join(RealEstate_path, sequence_name, f'{sequence_name}.txt')
                sequence_path = os.path.join(RealEstate_path, sequence_name, 'image_0')

                # Get a list of all poses [R,t] in this sequence
                poses = read_poses(specs_path)

                # Indices of 'good' image frames
                valid_indices = get_valid_indices(len(poses), sequence_path, jump_frames)
                if len(valid_indices) == 0: continue

                # Get projection matrix from calib.txt, compute intrinsic K, and adjust K according to transformations
                original_image_size = torch.tensor(Image.open(os.path.join(sequence_path, f'{valid_indices[0]:06}.{IMAGE_TYPE}')).size)
                K = get_intrinsic_REALESTATE(specs_path, original_image_size)
                
                if not FIRST_2_THRIDS_TRAIN and not FIRST_2_OF_3_TRAIN:
                    custom_dataset = Dataset(sequence_path, poses, valid_indices, transform, K, val=False, seq_name=sequence_name, jump_frames=jump_frames)
                    if len(custom_dataset) > 20:
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

def get_dataloaders_KITTI(batch_size=BATCH_SIZE):
    sequence_paths = [f'sequences/0{i}' for i in range(11)]
    poses_paths = [f'poses/0{i}.txt' for i in range(11)]
    calib_paths = [f'sequences/0{i}/calib.txt' for i in range(11)]

    train_datasets, val_datasets = [], []
    for i, (sequence_path, poses_path, calib_path) in enumerate(zip(sequence_paths, poses_paths, calib_paths)):
        if i not in train_seqeunces and i not in val_sequences: continue
        cam0_seq = os.path.join(sequence_path, 'image_0')
        cam1_seq = os.path.join(sequence_path, 'image_1')

        # Get a list of all poses [R,t] in this sequence
        poses = read_poses(poses_path)
        
        # Indices of 'good' image frames
        valid_indices = get_valid_indices(len(poses), cam0_seq)
    
        # Get projection matrix from calib.txt, compute intrinsic K, and adjust K according to transformations
        orginal_image_size = torch.tensor(Image.open(os.path.join(cam0_seq, f'{valid_indices[0]:06}.{IMAGE_TYPE}')).size)
        k0, k1 = get_intrinsic_KITTI(calib_path, orginal_image_size)

        # Split the dataset based on the calculated samples. Get 00 and 01 as val and the rest as train sets.
        dataset_cam0 = Dataset(cam0_seq, poses, valid_indices, transform, k0, val=False, seq_name= f'0{i}')
        dataset_cam1 = Dataset(cam1_seq, poses, valid_indices, transform, k1, val=True, seq_name= f'0{i}')
        if i in train_seqeunces:
            train_datasets.append(dataset_cam0)        
        if i in val_sequences:
            val_datasets.append(dataset_cam1)

    # Concatenate datasets
    concat_train_dataset = ConcatDataset(train_datasets)
    concat_val_dataset = ConcatDataset(val_datasets)

    # Create a DataLoader
    train_loader = DataLoader(concat_train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(concat_val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader


def get_data_loaders(batch_size=BATCH_SIZE):
    if USE_REALESTATE:
        return get_dataloaders_RealEstate(batch_size)
    else: # KITTI
        return get_dataloaders_KITTI(batch_size)
