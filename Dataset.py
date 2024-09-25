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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequence_path, poses, img0, img1, valid_indices, keypoints, transform, k, k_resized, seq_name, jump_frames=JUMP_FRAMES):
        self.sequence_path = sequence_path
        self.poses = poses
        self.images_0 = img0
        self.images_1 = img1
        self.transform = transform
        self.k = k
        self.k_resized = k_resized
        self.valid_indices = valid_indices
        self.keypoints = keypoints
        self.seq_name = seq_name
        self.jump_frames = jump_frames

    def __len__(self):
        return len(self.valid_indices) 

    def __getitem__(self, idx):
        idx = self.valid_indices[idx]

        img0 = self.images_0[idx] if INIT_DATA else torchvision.io.read_image(os.path.join(self.sequence_path, f'{idx:06}.{IMAGE_TYPE}'))
        img1 = self.images_1[idx] if INIT_DATA else torchvision.io.read_image(os.path.join(self.sequence_path, f'{idx+self.jump_frames:06}.{IMAGE_TYPE}'))

        H, W = img0.shape[1], img0.shape[2]

        # Gey keypoints on original image
        # unnormalized_F = get_F(self.k, self.k, self.poses, idx, self.jump_frames)
        # F = norm_layer(unnormalized_F.view(-1, 9)).view(3,3)
        # epi = EpipolarGeometry(img0, img1, F=F.unsqueeze(0), is_scaled=False)

        k=self.k_resized.clone()
        if RANDOM_CROP:
            top_crop, left_crop = random.randint(0, RESIZE-CROP), random.randint(0, RESIZE-CROP)
            img0, img1 = TF.resize(img0, (RESIZE, RESIZE), antialias=True), TF.resize(img1, (RESIZE, RESIZE), antialias=True)
            img0, img1 = TF.crop(img0, top_crop, left_crop, CROP, CROP), TF.crop(img1, top_crop, left_crop, CROP, CROP)
            k = adjust_k_crop(k, top_crop, left_crop)

        img0 = self.transform(img0) # shape (channels, height, width)
        img1 = self.transform(img1) # shape (channels, height, width)

        unnormalized_F = get_F(k, k, self.poses, idx, self.jump_frames)
        F = norm_layer(unnormalized_F.view(-1, 9)).view(3,3)

        # Adjust keypoints according to the crop
        # pts1, pts2 = adjust_points_no_dict(epi.pts1, epi.pts2, top_crop, left_crop, H, W)
        pts1, pts2 = adjust_points(self.keypoints, idx, top_crop, left_crop, height=H, width=W)

        return img0, img1, F, pts1, pts2, self.seq_name

class Dataset_stereo(torch.utils.data.Dataset):
    def __init__(self, sequence_path, transform, k0, k1, R, t, images_0, images_1, keypoints, subset_valid_indices, seq_name, test, data_ratio):
        self.sequence_path = sequence_path
        self.transform = transform
        self.k0 = k0
        self.k1 = k1
        self.R=R
        self.t=t        
        self.images_0 = images_0
        self.images_1 = images_1
        self.keypoints = keypoints
        self.subset_valid_indices = subset_valid_indices
        self.seq_name = seq_name
        self.test = test
        self.data_ratio = data_ratio

    def __len__(self):
        return int(len(self.subset_valid_indices))

    def __getitem__(self, idx):
        idx = self.subset_valid_indices[idx]

        img0 = self.images_0[idx] if INIT_DATA else torchvision.io.read_image(os.path.join(self.sequence_path, 'image_0', f'{idx:06}.{IMAGE_TYPE}')).to(device) # shape (channels, height, width)
        img1 = self.images_1[idx] if INIT_DATA else torchvision.io.read_image(os.path.join(self.sequence_path, 'image_1', f'{idx:06}.{IMAGE_TYPE}')).to(device) # shape (channels, height, width)
        H, W = img0.shape[1], img0.shape[2]

        k0=self.k0.clone()
        k1=self.k1.clone()
        if RANDOM_CROP:
            top_crop, left_crop = random.randint(0, RESIZE-CROP), random.randint(0, RESIZE-CROP)
            img0, img1 = TF.resize(img0, (RESIZE, RESIZE), antialias=True), TF.resize(img1, (RESIZE, RESIZE), antialias=True)
            img0, img1 = TF.crop(img0, top_crop, left_crop, CROP, CROP), TF.crop(img1, top_crop, left_crop, CROP, CROP)
            k0 = adjust_k_crop(k0, top_crop, left_crop)
            k1 = adjust_k_crop(k1, top_crop, left_crop)


        img0 = self.transform(img0) # shape (channels, height, width)
        img1 = self.transform(img1) # shape (channels, height, width)

        unnormalized_F = get_F(k0, k1, R_relative=self.R, t_relative=self.t)
        
        # Normalize F-Matrix
        F = norm_layer(unnormalized_F.view(-1, 9)).view(3,3)

        pts1, pts2 = adjust_points(self.keypoints, idx, top_crop, left_crop, height=H, width=W)
        
        return img0, img1, F, pts1, pts2, self.seq_name
    
def get_valid_indices(sequence_len, sequence_path, jump_frames=JUMP_FRAMES):
    valid_indices = []
    for idx in range(sequence_len - jump_frames):
        img0_path = os.path.join(sequence_path, f'{idx:06}.{IMAGE_TYPE}')
        img1_path = os.path.join(sequence_path, f'{idx+jump_frames:06}.{IMAGE_TYPE}')

        if os.path.exists(img0_path) and os.path.exists(img1_path):
            valid_indices.append(idx)

    return valid_indices


def get_transform():
    transforms = []
    if not RANDOM_CROP: # original image size ~ 1200 * 700
        transforms.extend([
            v2.Resize((RESIZE, RESIZE), antialias=True),
            v2.CenterCrop(CROP)
        ])
    transforms.append(v2.Grayscale(num_output_channels=3))
    if AUGMENTATION:
        transforms.append(v2.ColorJitter(brightness=0.3, contrast=0.3))
        transforms.append(v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.35)))
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
        if pts1.shape[0] == 0:
            print(f'\n################\nEmpty points at {seq_names}\n\n')
        pad_len = max_len - pts1.shape[0]
        padded_pts1.append(F.pad(pts1, (0, 0, 0, pad_len), 'constant', 0))
        padded_pts2.append(F.pad(pts2, (0, 0, 0, pad_len), 'constant', 0))  

    return (torch.stack(imgs1), torch.stack(imgs2), torch.stack(Fs), torch.stack(padded_pts1), torch.stack(padded_pts2), seq_names)

def get_dataloaders_RealEstate(train_num_sequences, batch_size):
    RealEstate_paths = ['RealEstate10K/train_images', 'RealEstate10K/val_images']
    train_datasets, val_datasets, test_datasets = [], [], []
    for jump_frames in [JUMP_FRAMES]:
        for RealEstate_path in RealEstate_paths:
            for i, sequence_name in enumerate(os.listdir(RealEstate_path)): 
                specs_path = os.path.join(RealEstate_path, sequence_name, f'{sequence_name}.txt')
                sequence_path = os.path.join(RealEstate_path, sequence_name, 'image_0')

                # Get a list of all poses [R,t] in this sequence
                poses = read_poses(specs_path).to(device)

                # Indices of 'good' image frames
                valid_indices = get_valid_indices(len(poses), sequence_path, jump_frames)

                # Get projection matrix from calib.txt, compute intrinsic K, and adjust K according to transformations
                original_image_size = torch.tensor(Image.open(os.path.join(sequence_path, f'{valid_indices[0]:06}.{IMAGE_TYPE}')).size).to(device)
                K = get_intrinsic_REALESTATE(specs_path, original_image_size, adjust_resize=False)
                K_resized = get_intrinsic_REALESTATE(specs_path, original_image_size, adjust_resize=True)

                img0 = {idx: torchvision.io.read_image(os.path.join(sequence_path, f'{idx:06}.{IMAGE_TYPE}')).to(device) for idx in valid_indices} if INIT_DATA else None    
                img1 = {idx: torchvision.io.read_image(os.path.join(sequence_path, f'{idx+jump_frames:06}.{IMAGE_TYPE}')).to(device) for idx in valid_indices} if INIT_DATA else None    

                keypoints_dict = load_keypoints(os.path.join(os.path.dirname(sequence_path), 'keypoints.txt'))

                custom_dataset = Dataset(sequence_path, poses, img0, img1, valid_indices, keypoints_dict, transform, K, K_resized, seq_name=sequence_name, jump_frames=jump_frames)

                if len(custom_dataset) > 9:
                    if RealEstate_path == 'RealEstate10K/train_images':
                        if len(train_datasets) > train_num_sequences: break                        
                        train_datasets.append(custom_dataset) 
                    elif sequence_name not in RL_TEST_NAMES:
                        val_datasets.append(custom_dataset)
                    else:
                        test_datasets.append(custom_dataset)
                else: 
                    print(f"Empty dataset at {RealEstate_path}, {sequence_name}: {len(custom_dataset)}")

    # Concatenate datasets
    concat_train_dataset = ConcatDataset(train_datasets)
    concat_val_dataset = ConcatDataset(val_datasets)
    concat_test_dataset = ConcatDataset(test_datasets)

    # Create a DataLoader
    train_loader = DataLoader(concat_train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False, collate_fn=custom_collate_fn)
    val_loader = DataLoader(concat_val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(concat_test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False, collate_fn=custom_collate_fn)
    
    print(len(train_loader), len(val_loader), len(test_loader))

    return train_loader, val_loader, test_loader

# def get_dataloaders_KITTI(data_ratio, batch_size):
#     sequence_paths = [f'sequences/0{i}' for i in range(11)]
#     poses_paths = [f'poses/0{i}.txt' for i in range(11)]
#     calib_paths = [f'sequences/0{i}/calib.txt' for i in range(11)]

#     train_datasets, val_datasets = [], []
#     for jump_frames in [JUMP_FRAMES]:
#         for i, (sequence_path, poses_path, calib_path) in enumerate(zip(sequence_paths, poses_paths, calib_paths)):
#             if i not in train_seqeunces and i not in val_sequences: continue
#             cam0_seq = os.path.join(sequence_path, 'image_0')
#             cam1_seq = os.path.join(sequence_path, 'image_1')

#             # Get a list of all poses [R,t] in this sequence
#             poses = read_poses(poses_path)
            
#             # Indices of 'good' image frames
#             valid_indices = get_valid_indices(len(poses), cam0_seq, jump_frames)
        
#             # Get projection matrix from calib.txt, compute intrinsic K, and adjust K according to transformations
#             original_image_size = torch.tensor(Image.open(os.path.join(cam0_seq, f'{valid_indices[0]:06}.{IMAGE_TYPE}')).size)
#             k0, k1 = get_intrinsic_KITTI(calib_path, original_image_size)

#             # Split the dataset based on the calculated samples. Get 00 and 01 as val and the rest as train sets.
#             dataset_cam0 = Dataset(cam0_seq, poses, valid_indices, transform, k0, val=False, seq_name= f'0{i}', jump_frames=jump_frames)
#             dataset_cam1 = Dataset(cam1_seq, poses, valid_indices, transform, k1, val=True, seq_name= f'0{i}', jump_frames=jump_frames)
#             if i in train_seqeunces:
#                 train_datasets.append(dataset_cam0)        
#             if i in val_sequences:
#                 val_datasets.append(dataset_cam1)

#     # Concatenate datasets
#     concat_train_dataset = ConcatDataset(train_datasets)
#     concat_val_dataset = ConcatDataset(val_datasets)

#     # Create a DataLoader
#     train_loader = DataLoader(concat_train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
#     val_loader = DataLoader(concat_val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

#     return train_loader, val_loader

def get_dataloader_stereo(data_ratio, part, batch_size, num_workers=NUM_WORKERS):
    sequence_paths = [f'sequences/{i:02}' for i in range(11)]
    poses_paths = [f'poses/{i:02}.txt' for i in range(11)]
    calib_paths = [f'sequences/{i:02}/calib.txt' for i in range(11)] 
      
    R_relative = torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype=torch.float32).to(device)
    t_relative = torch.tensor([0.54, 0, 0], dtype=torch.float32).to(device)

    train_datasets, val_datasets, test_datasets = [], [], []
    for i, (sequence_path, poses_path, calib_path) in enumerate(zip(sequence_paths, poses_paths, calib_paths)):
        if i not in train_seqeunces_stereo and i not in val_sequences_stereo and i not in test_sequences_stereo: continue
        
        image_0_path = os.path.join(sequence_path, 'image_0')

        # Get a list of all poses [R,t] in this sequence
        poses = read_poses(poses_path)
        
        # Indices of 'good' image frames
        valid_indices = get_valid_indices(len(poses), image_0_path, jump_frames=0)

        # Get projection matrix from calib.txt, compute intrinsic K, and adjust K according to transformations
        original_image_size = torch.tensor(Image.open(os.path.join(image_0_path, f'{valid_indices[0]:06}.{IMAGE_TYPE}')).size).to(device)
        k0, k1 = get_intrinsic_KITTI(calib_path, original_image_size)
        
        if i in test_sequences_stereo:
            subset = valid_indices
        else:
            length = int(len(valid_indices) * data_ratio) 
            mid_start = len(valid_indices) // 2 - length // 2
            subset = valid_indices[:length] if part == "head" else valid_indices[mid_start:mid_start+length] if part == "mid" else valid_indices[-length:] if part == "tail" else None

        images_0 = {idx: torchvision.io.read_image(os.path.join(sequence_path, 'image_0', f'{idx:06}.{IMAGE_TYPE}')).to(device) for idx in subset} if INIT_DATA else None    
        images_1 = {idx: torchvision.io.read_image(os.path.join(sequence_path, 'image_1', f'{idx:06}.{IMAGE_TYPE}')).to(device) for idx in subset} if INIT_DATA else None

        keypoints_dict = load_keypoints(os.path.join(sequence_path, 'keypoints.txt'))

        dataset_stereo = Dataset_stereo(sequence_path, transform, k0, k1, R_relative, t_relative, images_0, images_1, keypoints_dict, subset, seq_name= f'0{i}', test=True if i in test_sequences_stereo else False, data_ratio=data_ratio)

        if i in train_seqeunces_stereo:
            train_datasets.append(dataset_stereo)        
        if i in val_sequences_stereo:
            val_datasets.append(dataset_stereo)
        if i in test_sequences_stereo:
            test_datasets.append(dataset_stereo)

    # Concatenate datasets
    concat_train_dataset = ConcatDataset(train_datasets)
    concat_val_dataset = ConcatDataset(val_datasets)
    concat_test_dataset = ConcatDataset(test_datasets)

    # Create a DataLoader
    train_loader = DataLoader(concat_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=custom_collate_fn)
    val_loader = DataLoader(concat_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn)
    test_loader = DataLoader(concat_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn)

    return train_loader, val_loader, test_loader

def get_data_loaders(train_size=None, part=None, batch_size=BATCH_SIZE):
    if STEREO:
        return get_dataloader_stereo(train_size, part, batch_size)
    elif USE_REALESTATE:
        return get_dataloaders_RealEstate(train_size, batch_size)
    else: # KITTI
        # return get_dataloaders_KITTI(train_size, batch_size)
        return None


def save_keypoints_stereo():
    """
    To work put these lines in the init of EpipolarGeometry:
        self.image1_numpy = image1_tensors.cpu().numpy().transpose(1, 2, 0)
        self.image2_numpy = image2_tensors.cpu().numpy().transpose(1, 2, 0) 
    """
    sequence_paths = [f'sequences/{i:02}' for i in range(11)]
    poses_paths = [f'poses/{i:02}.txt' for i in range(11)]
    calib_paths = [f'sequences/{i:02}/calib.txt' for i in range(11)]  
      
    R = torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype=torch.float32).to(device)
    t = torch.tensor([0.54, 0, 0], dtype=torch.float32).to(device)

    for i, (sequence_path, poses_path, calib_path) in enumerate(zip(sequence_paths, poses_paths, calib_paths)):
        if i not in train_seqeunces_stereo and i not in val_sequences_stereo and i not in test_sequences_stereo: continue

        image_0_path = os.path.join(sequence_path, 'image_0')

        # Get a list of all poses [R,t] in this sequence
        poses = read_poses(poses_path)
        
        # Indices of 'good' image frames
        valid_indices = get_valid_indices(len(poses), image_0_path, jump_frames=0)

        # Get projection matrix from calib.txt, compute intrinsic K, and adjust K according to transformations
        original_image_size = torch.tensor(Image.open(os.path.join(image_0_path, f'{valid_indices[0]:06}.{IMAGE_TYPE}')).size).to(device)
        k0, k1 = get_intrinsic_KITTI(calib_path, original_image_size, adjust_resize=False)

        for idx in valid_indices:
            img0 = torchvision.io.read_image(os.path.join(sequence_path, 'image_0', f'{idx:06}.{IMAGE_TYPE}')) # shape (channels, height, width)
            img1 = torchvision.io.read_image(os.path.join(sequence_path, 'image_1', f'{idx:06}.{IMAGE_TYPE}')) # shape (channels, height, width)
            
            img0 = TF.rgb_to_grayscale(img0, num_output_channels=1)
            img1 = TF.rgb_to_grayscale(img1, num_output_channels=1)

            unnormalized_F = get_F(k0, k1, R_relative=R, t_relative=t)
            
            # Normalize F-Matrix
            F = norm_layer(unnormalized_F.view(-1, 9)).view(3,3)

            epi = EpipolarGeometry(img0, img1, F=F.unsqueeze(0), is_scaled=False)

            filepath = os.path.join(sequence_path, f'keypoints.txt')
            with open(filepath, 'a') as f:
                line = f"{idx}; {epi.pts1.tolist()}; {epi.pts2.tolist()}\n"
                f.write(line)


            # Convert grayscale tensors to numpy arrays for matplotlib
            # img0_np = img0.squeeze().numpy()
            # img1_np = img1.squeeze().numpy()

            # # Create a subplot with two images
            # fig, axs = plt.subplots(1, 2, figsize=(15, 7))

            # # Display the first image with keypoints
            # axs[0].imshow(img0_np, cmap='gray')
            # axs[0].scatter(epi.pts1[:, 0].numpy(), epi.pts1[:, 1].numpy(), c='r', s=10)
            # axs[0].set_title(f'Image 0 - {idx}')
            # axs[0].axis('off')

            # # Display the second image with keypoints
            # axs[1].imshow(img1_np, cmap='gray')
            # axs[1].scatter(epi.pts2[:, 0].numpy(), epi.pts2[:, 1].numpy(), c='r', s=10)
            # axs[1].set_title(f'Image 1 - {idx}')
            # axs[1].axis('off')

            # # Show the plot
            # plt.show()


def save_keypoints_realestate():
    RealEstate_paths = ['RealEstate10K/val_images']
    for jump_frames in [JUMP_FRAMES]:
        for RealEstate_path in RealEstate_paths:
            for i, sequence_name in enumerate(os.listdir(RealEstate_path)): 
                specs_path = os.path.join(RealEstate_path, sequence_name, f'{sequence_name}.txt')
                sequence_path = os.path.join(RealEstate_path, sequence_name, 'image_0')

                # Get a list of all poses [R,t] in this sequence
                poses = read_poses(specs_path).to(device)

                # Indices of 'good' image frames
                valid_indices = get_valid_indices(len(poses), sequence_path, jump_frames)

                # Get projection matrix from calib.txt, compute intrinsic K, and adjust K according to transformations
                original_image_size = torch.tensor(Image.open(os.path.join(sequence_path, f'{valid_indices[0]:06}.{IMAGE_TYPE}')).size).to(device)
                k = get_intrinsic_REALESTATE(specs_path, original_image_size, adjust_resize=False)
                original_image_size = torch.tensor(Image.open(os.path.join(sequence_path, f'{valid_indices[0]:06}.{IMAGE_TYPE}')).size).to(device)

                for idx in valid_indices:
                    a = torch.tensor(Image.open(os.path.join(sequence_path, f'{valid_indices[idx]:06}.{IMAGE_TYPE}')).size).to(device)
                    print(a, original_image_size)
                    if a != original_image_size:
                        print(f"Different sizes: {a}, {original_image_size}")
                    # img0 = torchvision.io.read_image(os.path.join(sequence_path, f'{idx:06}.{IMAGE_TYPE}'))
                    # img1 = torchvision.io.read_image(os.path.join(sequence_path, f'{idx+jump_frames:06}.{IMAGE_TYPE}'))
                    
                    # img0 = TF.rgb_to_grayscale(img0, num_output_channels=1)
                    # img1 = TF.rgb_to_grayscale(img1, num_output_channels=1)

                    # # Gey keypoints on original image
                    # unnormalized_F = get_F(k, k, poses, idx, jump_frames)
                    
                    # # Normalize F-Matrix
                    # F = norm_layer(unnormalized_F.view(-1, 9)).view(3,3)

                    # epi = EpipolarGeometry(img0, img1, F=F.unsqueeze(0), is_scaled=False)

                    # # filepath = os.path.join(os.path.dirname(sequence_path), f'keypoints.txt')
                    # # with open(filepath, 'a') as f:
                    # #     line = f"{idx}; {epi.pts1.tolist()}; {epi.pts2.tolist()}\n"
                    # #     f.write(line)


                    # # Convert grayscale tensors to numpy arrays for matplotlib
                    # img0_np = img0.squeeze().numpy()
                    # img1_np = img1.squeeze().numpy()

                    # Create a subplot with two images
                    # fig, axs = plt.subplots(1, 2, figsize=(15, 7))

                    # # Display the first image with keypoints
                    # axs[0].imshow(img0_np, cmap='gray')
                    # axs[0].scatter(epi.pts1[:, 0].numpy(), epi.pts1[:, 1].numpy(), c='r', s=10)
                    # axs[0].set_title(f'Image 0 - {idx}')
                    # axs[0].axis('off')

                    # # Display the second image with keypoints
                    # axs[1].imshow(img1_np, cmap='gray')
                    # axs[1].scatter(epi.pts2[:, 0].numpy(), epi.pts2[:, 1].numpy(), c='r', s=10)
                    # axs[1].set_title(f'Image 1 - {idx}')
                    # axs[1].axis('off')

                    # Show the plot
                    # plt.show()
                    # plt.savefig(f'keypoints_{sequence_name}_{idx}.png')


                    # Draw keypoints on image 0
                    # for i,(pt) in enumerate(epi.pts1.cpu().numpy()):
                    #     if i == 20:break
                    #     cv2.circle(img0_np, (int(pt[0]), int(pt[1])), radius=5, color=(255, 255, 0), thickness=-1)

                    # # Draw keypoints on image 1
                    # for i,(pt) in enumerate(epi.pts2.cpu().numpy()):
                    #     if i == 20:break
                    #     cv2.circle(img1_np, (int(pt[0]), int(pt[1])), radius=5, color=(255, 255, 0), thickness=-1)

                    # # Save the images
                    # os.makedirs('draw0', exist_ok=True)
                    # combined_image = np.hstack((img0_np, img1_np))

                    # cv2.imwrite(f'draw0/images_with_keypoints_{idx}.png', combined_image)
                    # print("Saved images")

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    save_keypoints_realestate()    