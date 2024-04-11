import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from FunMatrix import *
from utils import *

from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import v2

from PIL import Image
import torchvision


class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequence_path, poses, valid_indices, transform, K):
        self.sequence_path = sequence_path
        self.sequence_num = sequence_path.split('/')[1]
        self.poses = poses
        self.transform = transform
        self.k = K
        self.valid_indices = valid_indices

    def __len__(self):
        return len(self.valid_indices) - JUMP_FRAMES

    def __getitem__(self, idx):
        idx = self.valid_indices[idx]
        
        original_first_image = torchvision.io.read_image(os.path.join(self.sequence_path, f'{idx:06}.{IMAGE_TYPE}'))
        original_second_image = torchvision.io.read_image(os.path.join(self.sequence_path, f'{idx+JUMP_FRAMES:06}.{IMAGE_TYPE}'))

        # Transform: Resize, center, grayscale
        first_image = self.transform(original_first_image)
        second_image = self.transform(original_second_image)

        unnormalized_F = get_F(self.poses, idx, self.k)

        # Normalize F-Matrix
        F = norm_layer(unnormalized_F.view(-1, 9)).view(3,3)

        return first_image, second_image, F

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
        v2.Resize((256, 256)),
        v2.CenterCrop(224),
        v2.Grayscale(num_output_channels=3),
        v2.ColorJitter(brightness=(0.85, 1.15), contrast=(0.85, 1.15)),
        v2.ToDtype(torch.float32, scale=True),  # Converts to torch.float32 and scales [0,255] -> [0,1]
        v2.Normalize(mean=norm_mean,  # Normalize each channel
                            std=norm_std),
    ])    
else:
    transform = v2.Compose([
        v2.Resize((256, 256)),
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
            
            # Get projection matrix from calib.txt, compute intrinsic K, and adjust K according to transformations
            original_image_size = torch.tensor(Image.open(os.path.join(sequence_path, f'{valid_indices[0]:06}.{IMAGE_TYPE}')).size)
            K = get_intrinsic_REALESTATE(specs_path, original_image_size)
            
            custom_dataset = Dataset(sequence_path, poses, valid_indices, transform, K)
            if len(custom_dataset) > 30:
                if RealEstate_path == 'RealEstate10K/train_images':
                    train_datasets.append(custom_dataset) 
                else:    
                    val_datasets.append(custom_dataset)
                
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
    train_loader = DataLoader(concat_train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(concat_val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

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


# Function to denormalize image
def denormalize(image, mean, std):
    image = image.clone().numpy().transpose((1, 2, 0))  # Change from (C, H, W) to (H, W, C)
    mean = np.array(mean)
    std = np.array(std)
    image = std * image + mean  # Denormalize
    image = np.clip(image, 0, 1)  # Clip to [0, 1] range
    return image

# Function to visualize a batch of images
def show_images(first_image, second_image):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # Denormalize and visualize the first image
    ax[0].imshow(denormalize(first_image[0], norm_mean, norm_std))
    ax[0].set_title('First Image')
    ax[0].axis('off')

    # Denormalize and visualize the second image
    ax[1].imshow(denormalize(second_image[0], norm_mean, norm_std))
    ax[1].set_title('Second Image')
    ax[1].axis('off')

    plt.show()

if __name__ == "__main__":
    # Get the train_loader
    train_loader, _ = get_data_loaders(batch_size=1)
    i=0
    for img_1,img_2,label in train_loader:
        epipolar_geo_pred = EpipolarGeometry(img_1[0],img_2[0], label[0]) 
        plots_path = 'epipolar_lines'
        epipolar_geo_pred.visualize(sqResultDir=os.path.join(plots_path), file_num=i)
        i = i + 1
    
    # Iterate over the train_loader
    # for first_image, second_image, _ in train_loader:
    #     show_images(first_image, second_image)
        
    #     # Break or wait for user input to continue showing images
    #     input("Press Enter to continue...")  # Wait for user input to continue
    #     # If you want to break after the first batch, uncomment the following line
    #     # break