from FunMatrix import *
from utils import *
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as T


 
class CustomDataset_first_two_thirds_train(torch.utils.data.Dataset):
    """Takes the first 2/3 images in the sequence for training, and the last 1/3 for testing"""

    def __init__(self, sequence_path, poses, valid_indices, transform, K, dataset_type):
        self.sequence_path = sequence_path
        self.sequence_num = sequence_path.split('/')[1]
        self.poses = poses
        self.transform = transform
        self.k = K
        self.valid_indices = valid_indices
        self.dataset_type = dataset_type

    def __len__(self):
            # Adjust the total count based on dataset type
            if self.dataset_type == 'train':
                return ((len(self.valid_indices)-JUMP_FRAMES) // 3) * 2  # 2/3 if the set
            else:
                return (len(self.valid_indices)-JUMP_FRAMES) // 3  # 1/3 if the set
            
    def __getitem__(self, idx):
        try: 
            if self.dataset_type == 'train':
                idx = self.valid_indices[idx]
            else:
                idx = self.valid_indices[idx + ((len(self.valid_indices) - JUMP_FRAMES) // 3) * 2]

            original_first_image = Image.open(os.path.join(self.sequence_path, f'{idx:06}.{IMAGE_TYPE}'))
            original_second_image = Image.open(os.path.join(self.sequence_path, f'{idx+JUMP_FRAMES:06}.{IMAGE_TYPE}'))
        except Exception as e:
            print_and_write(f"1\nError in sequence: {self.sequence_path}, idx: {idx}, dataset_type: {self.dataset_type} sequence num: {self.sequence_num}\nException: {e}")
        
        try:
            # Transform: Resize, center, grayscale
            first_image = transform(original_first_image)
            second_image = transform(original_second_image)
            # first_image = self.transform(original_first_image)
            # second_image = self.transform(original_second_image)
        except Exception as e:
            print_and_write(f"2\nError in sequence: {self.sequence_path}, idx: {idx}, dataset_type: {self.dataset_type} sequence num: {self.sequence_num}\nException: {e}")
        
        try:
            first_image = first_image.to(device)
            second_image = second_image.to(device)
        except Exception as e:
            print_and_write(f"3\nError in sequence: {self.sequence_path}, idx: {idx}, dataset_type: {self.dataset_type} sequence num: {self.sequence_num}\nException: {e}")
        
        try:
            if PREDICT_POSE:
                unormalized_R, unormalized_t = compute_relative_transformations(self.poses[idx],self. poses[idx+JUMP_FRAMES])
                unormalized_label = torch.cat((unormalized_R, unormalized_t.view(3,1)), dim=-1)
            else:
                unormalized_label = get_F(self.poses, idx, self.k)
        except Exception as e:
            print_and_write(f"4\nError in sequence: {self.sequence_path}, idx: {idx}, dataset_type: {self.dataset_type} sequence num: {self.sequence_num}\nException: {e}")
        try:
            if PREDICT_POSE:
                R, t = norm_layer(unormalized_R.view(-1, 9)).view(3,3), norm_layer(unormalized_t.view(-1, 3), predict_t=True).view(3)

                label = torch.cat((R, t.view(3,1)), dim=-1)
            else:               
                label = norm_layer(unormalized_label.view(-1, 9)).view(3,3)
        except Exception as e:
            print_and_write("5\n {e}")
            
        return first_image, second_image, label, unormalized_label, self.k
    
class CustomDataset_first_two_out_of_three_train(torch.utils.data.Dataset):
    """Takes the first two images out of every three images in the sequence for training, and the third for testing"""

    def __init__(self, sequence_path, poses, valid_indices, transform, K, dataset_type):
        self.sequence_path = sequence_path
        self.sequence_num = sequence_path.split('/')[1]
        self.poses = poses
        self.transform = transform
        self.k = K
        self.valid_indices = valid_indices
        self.dataset_type = dataset_type

    def __len__(self):
            # Adjust the total count based on dataset type
            if self.dataset_type == 'train':
                return ((len(self.valid_indices)-JUMP_FRAMES) // 3) * 2  # 2 out of every 3 images
            else:
                return (len(self.valid_indices)-JUMP_FRAMES) // 3  # Every 3rd image
            
    def __getitem__(self, idx):
        try: 
            if self.dataset_type == 'train':
            # Map idx to include 2 out of every 3 images
                idx = idx + (idx // 2)
            else:
            # Map idx to select every 3rd image
                idx = idx * 3 + 2

            original_first_image = Image.open(os.path.join(self.sequence_path, f'{idx:06}.{IMAGE_TYPE}'))
            original_second_image = Image.open(os.path.join(self.sequence_path, f'{idx+JUMP_FRAMES:06}.{IMAGE_TYPE}'))
        except Exception as e:
            print_and_write(f"1\nError in sequence: {self.sequence_path}, idx: {idx}, dataset_type: {self.dataset_type} sequence num: {self.sequence_num}\nException: {e}")
        
        try:
            # Transform: Resize, center, grayscale
            first_image = transform(original_first_image).to(device)
            second_image = transform(original_second_image).to(device)
            # first_image = self.transform(original_first_image).to(device)
            # second_image = self.transform(original_second_image).to(device)
        except Exception as e:
            print_and_write(f"2\nError in sequence: {self.sequence_path}, idx: {idx}, dataset_type: {self.dataset_type} sequence num: {self.sequence_num}\nException: {e}")
        
        try:
            unnormalized_F = get_F(self.poses, idx, self.k)
        except Exception as e:
            print_and_write(f"3\nError in sequence: {self.sequence_path}, idx: {idx}, dataset_type: {self.dataset_type} sequence num: {self.sequence_num}\nException: {e}")
        
        try:
            # Normalize F-Matrix
            F = norm_layer(unnormalized_F.view(-1, 9)).view(3,3)
        except Exception as e:
            print_and_write(f"4\nError in sequence: {self.sequence_path}, idx: {idx}, dataset_type: {self.dataset_type} sequence num: {self.sequence_num}\nException: {e}")
        
        
        return first_image, second_image, F, unnormalized_F

def transform(img):
    try:
        resized_image = transforms.Resize((256, 256))(img)
    except Exception as e:
        print_and_write(f"5/resized {e}")

    try:
        cropped_image = transforms.CenterCrop(224)(resized_image)
    except Exception as e:
        print_and_write(f"6/cropped {e}")

    try:
        grayscale_image = transforms.Grayscale(num_output_channels=3)(cropped_image)
    except Exception as e:
        print_and_write(f"7/grayscale {e}")

    try:
        tensor_image = transforms.ToTensor()(grayscale_image)
    except Exception as e:
        print_and_write(f"8/tensor {e}")

    try:
        normalized_image = transforms.Normalize(mean=norm_mean, std=norm_std)(tensor_image)
    except Exception as e:
        print_and_write(f"9/normalize {e}")

    return normalized_image

def get_valid_indices(sequence_len, sequence_path):
    valid_indices = []
    for idx in range(sequence_len - JUMP_FRAMES):
        img1_path = os.path.join(sequence_path, f'{idx:06}.{IMAGE_TYPE}')
        img2_path = os.path.join(sequence_path, f'{idx+JUMP_FRAMES:06}.{IMAGE_TYPE}')

        if os.path.exists(img1_path) and os.path.exists(img2_path):
            valid_indices.append(idx)

    return valid_indices

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),                # Converts to tensor and rescales [0,255] -> [0,1]
    transforms.Normalize(mean=norm_mean,  # Normalize each channel
                         std=norm_std),
])    

def data_with_one_sequence(batch_size, CustomDataset_type):
    RealEstate_path = 'RealEstate10K/train_images'
    # sequence_name = '0cb8672999a42a05'
    sequence_name = "0000cc6d8b108390"

    specs_path = os.path.join(RealEstate_path, sequence_name, f'{sequence_name}.txt')
    sequence_path = os.path.join(RealEstate_path, sequence_name, 'image_0')

    # Get a list of all poses [R,t] in this sequence
    poses = read_poses(specs_path).to(device)

    # Indices of 'good' image frames
    valid_indices = get_valid_indices(len(poses), sequence_path)
    
    # Get projection matrix from calib.txt, compute intrinsic K, and adjust K according to transformations
    original_image_size = torch.tensor(Image.open(os.path.join(sequence_path, f'{valid_indices[0]:06}.{IMAGE_TYPE}')).size).to(device)
    K = get_intrinsic_REALESTATE(specs_path, original_image_size)
    
    train_dataset = CustomDataset_first_two_thirds_train(sequence_path, poses, valid_indices, transform, K, dataset_type='train')
    val_dataset = CustomDataset_first_two_thirds_train(sequence_path, poses, valid_indices, transform, K, dataset_type='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader

def data_for_checking_overfit(batch_size, CustomDataset_type):
    RealEstate_paths = ['RealEstate10K/train_images', 'RealEstate10K/val_images']

    train_datasets, val_datasets = [], []
    for RealEstate_path in RealEstate_paths:
        for i, sequence_name in enumerate(os.listdir(RealEstate_path)):
            specs_path = os.path.join(RealEstate_path, sequence_name, f'{sequence_name}.txt')
            sequence_path = os.path.join(RealEstate_path, sequence_name, 'image_0')

            # Get a list of all poses [R,t] in this sequence
            poses = read_poses(specs_path).to(device)

            # Indices of 'good' image frames
            valid_indices = get_valid_indices(len(poses), sequence_path)
            
            # Get projection matrix from calib.txt, compute intrinsic K, and adjust K according to transformations
            original_image_size = torch.tensor(Image.open(os.path.join(sequence_path, f'{valid_indices[0]:06}.{IMAGE_TYPE}')).size).to(device)
            K = get_intrinsic_REALESTATE(specs_path, original_image_size)
            
            if CustomDataset_type == "CustomDataset_first_two_out_of_three_train":
                train_dataset = CustomDataset_first_two_out_of_three_train(sequence_path, poses, valid_indices, transform, K, dataset_type='train')
                val_dataset = CustomDataset_first_two_out_of_three_train(sequence_path, poses, valid_indices, transform, K, dataset_type='val')
            elif  CustomDataset_type == "CustomDataset_first_two_thirds_train":
                train_dataset = CustomDataset_first_two_thirds_train(sequence_path, poses, valid_indices, transform, K, dataset_type='train')
                val_dataset = CustomDataset_first_two_thirds_train(sequence_path, poses, valid_indices, transform, K, dataset_type='val')

            if len(val_dataset) > 30:
                train_datasets.append(train_dataset)     
                val_datasets.append(val_dataset)
                
    # Concatenate datasets
    concat_train_dataset = ConcatDataset(train_datasets)
    concat_val_dataset = ConcatDataset(val_datasets)

    # Create a DataLoader
    train_loader = DataLoader(concat_train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(concat_val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader

def add_noise_to_F(F, noise_level):
    noise = torch.randn_like(F) * noise_level
    print(torch.mean(torch.abs(noise)))
    return F + noise

def make_rank_2(F):
        U1, S1, Vt1 = torch.linalg.svd(F, full_matrices=False)

        S1[-1] = 0

        output = torch.matmul(torch.matmul(U1, torch.diag_embed(S1)), Vt1)

        if torch.linalg.matrix_rank(output) != 2:
            print(f'rank of ground-truth not 2: {torch.linalg.matrix_rank(F)}')
        return output

# if __name__ == "__main__":
#     train_loader, val_loader = data_with_one_sequence(batch_size=1,CustomDataset_type=CUSTOMDATASET_TYPE)
    
#     avg_ep_err_unnormalized, avg_ep_err = 0, 0
#     for i,(first_image, second_image, label, unormalized_label,_) in enumerate(val_loader):
#         batch_ep_err_unnormalized, batch_ep_err = 0, 0
#         for img_1, img_2, F, unormalized_F in zip(first_image, second_image, label, unormalized_label):
#             batch_ep_err_unnormalized += EpipolarGeometry(img_1, img_2, unormalized_F).get_epipolar_err()
#             batch_ep_err += EpipolarGeometry(img_1, img_2, F).get_epipolar_err()
#             print(torch.mean(F), torch.mean(unormalized_F))
#             # os.makedirs(os.path.join('unormalized'), exist_ok=True)
#             # os.makedirs(os.path.join('normalized'), exist_ok=True)

#             # epipolar_geo_unormalized = EpipolarGeometry(first_image[0], second_image[0], F=unormalized_label)
#             # epipolar_geo_unormalized.visualize(sqResultDir='unormalized', file_num=i)

#             # epipolar_geo = EpipolarGeometry(first_image[0], second_image[0], F=label)
#             # epipolar_geo.visualize(sqResultDir='normalized', file_num=i)
            
#         batch_ep_err_unnormalized, batch_ep_err = batch_ep_err_unnormalized/len(first_image), batch_ep_err/len(first_image)
#         avg_ep_err_unnormalized, avg_ep_err = avg_ep_err_unnormalized + batch_ep_err_unnormalized, avg_ep_err + batch_ep_err

#     avg_ep_err_unnormalized, avg_ep_err = avg_ep_err_unnormalized/len(val_loader), avg_ep_err/len(val_loader)
#     print(avg_ep_err_unnormalized, avg_ep_err)

