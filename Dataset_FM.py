import random
from FunMatrix import *
from utils import *
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import v2
import os
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F


class Dataset_FM(torch.utils.data.Dataset):
    def __init__(self, transform, images_0, images_1, Fs, seq_name):
        self.transform = transform       
        self.images_0 = images_0
        self.images_1 = images_1
        self.Fs = Fs
        self.seq_name = seq_name

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        idx = idx + 1
        F = self.Fs[idx]
        img0 = self.images_0[idx] # shape (channels, height, width)
        img1 = self.images_1[idx] # shape (channels, height, width)


        img0 = self.transform(img0) # shape (channels, height, width)
        img1 = self.transform(img1) # shape (channels, height, width)
        print(img0.shape)
        epi = EpipolarGeometry(img0, img1, F=F) 
        
        return img0, img1, F, epi.pts1, epi.pts2, self.seq_name
    


def get_transform():
    transforms = []
    
    # if not RANDOM_CROP: # original image size ~ 1200 * 700
    #     transforms.extend([
    #         v2.Resize((RESIZE, RESIZE), antialias=True),
    #         v2.CenterCrop(CROP)
    #     ])
    transforms.append(v2.Grayscale(num_output_channels=3))
    # if AUGMENTATION:
    #     transforms.append(v2.ColorJitter(brightness=0.3, contrast=0.3))
    #     transforms.append(v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.35)))
    transforms.append(v2.ToDtype(torch.float32, scale=True)) # Converts to torch.float32 and scales [0,255] -> [0,1]
    transforms.append(v2.Normalize(mean=norm_mean.to(device), std=norm_std.to(device))),  # Normalize each channel
    
    return v2.Compose(transforms)
transform = get_transform()    

def custom_collate_fn(batch):
    imgs1, imgs2, Fs, all_pts1, all_pts2, seq_names = zip(*batch)

    max_len = max(pts1.shape[0] for pts1 in all_pts1)

    padded_pts1 = []
    padded_pts2 = []
    for pts1, pts2 in zip(all_pts1, all_pts2):
        pad_len = max_len - pts1.shape[0]
        padded_pts1.append(F.pad(pts1, (0, 0, 0, pad_len), 'constant', 0))
        padded_pts2.append(F.pad(pts2, (0, 0, 0, pad_len), 'constant', 0))  

    return (torch.stack(imgs1), torch.stack(imgs2), torch.stack(Fs), torch.stack(padded_pts1), torch.stack(padded_pts2), seq_names)


def get_dataloader_FM(batch_size, num_workers=NUM_WORKERS):
    gt_paths = ["FM_Net/head_F/F", "FM_Net/leg_F/F", "FM_Net/pelvis_F/F", "FM_Net/spine_F/F"]
    pa_paths = ["FM_Net/head_generator_0/generator_0", "FM_Net/leg_generator_0/generator_0", "FM_Net/pelvis_generator_0/generator_0", "FM_Net/spine_generator_0/generator_0"]
    lat_paths = ["FM_Net/head_generator_90/generator_90", "FM_Net/leg_generator_90/generator_90", "FM_Net/pelvis_generator_90/generator_90", "FM_Net/spine_generator_90/generator_90"]

    train_datasets, val_datasets, test_datasets = [], [], []
    for i, (gt_path, pa_path, lat_path) in enumerate(zip(gt_paths, pa_paths, lat_paths)):
        Fs = read_F_FM(gt_path).to(device)
        
        valid_indices = range(1, 101)


        images_0 = {idx: torchvision.io.read_image(os.path.join(pa_path, f'{idx:}.jpg')).to(device) for idx in valid_indices} if INIT_DATA else None    
        images_1 = {idx: torchvision.io.read_image(os.path.join(lat_path, f'{idx}.jpg')).to(device) for idx in valid_indices} if INIT_DATA else None

        dataset_FM = Dataset_FM(transform, images_0, images_1, Fs, seq_name=f"seq_{i}")

        train_datasets.append(dataset_FM)  
    
    concat_train_dataset = ConcatDataset(train_datasets)
    
    train_loader = DataLoader(concat_train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn)

    return train_loader

