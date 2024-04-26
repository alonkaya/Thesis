import shutil
import signal
import sys
from FunMatrix import *
from utils import *
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import os
from PIL import Image
import traceback
import torchvision
 
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
        if self.dataset_type == 'train':
            idx = self.valid_indices[idx]
        else:
            idx = self.valid_indices[idx + ((len(self.valid_indices) - JUMP_FRAMES) // 3) * 2]
            
        original_first_image = torchvision.io.read_image(os.path.join(self.sequence_path, f'{idx:06}.{IMAGE_TYPE}'))
        original_second_image = torchvision.io.read_image(os.path.join(self.sequence_path, f'{idx+JUMP_FRAMES:06}.{IMAGE_TYPE}'))

        first_image = self.transform(original_first_image)
        second_image = self.transform(original_second_image)

        unnormalized_F = get_F(self.poses, idx, self.k)

        # Normalize F-Matrix
        F = norm_layer(unnormalized_F.view(-1, 9)).view(3,3)

        return first_image, second_image, F

 
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
        if self.dataset_type == 'train':
        # Map idx to include 2 out of every 3 images
            idx = idx + (idx // 2)
        else:
        # Map idx to select every 3rd image
            idx = idx * 3 + 2

        original_first_image = torchvision.io.read_image(os.path.join(self.sequence_path, f'{idx:06}.{IMAGE_TYPE}'))
        original_second_image = torchvision.io.read_image(os.path.join(self.sequence_path, f'{idx+JUMP_FRAMES:06}.{IMAGE_TYPE}'))

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


def worker_init_fn(worker_id):
    def signal_handler(signal, frame):
        print(f'Worker {worker_id} received signal, exiting gracefully.')
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    def worker_exception_handler(exception_type, exception, traceback_details):
        # Here you can log the exception in a way that suits you
        print(f"Worker {worker_id} exception: {exception}, Traceback: {traceback.format_tb(traceback_details)}")

    sys.excepthook = worker_exception_handler


def data_with_one_sequence(batch_size, sequence_name='f1ee9dc6135e5307'):
    RealEstate_path = 'RealEstate10K/val_images'
    # sequence_name = '0cb8672999a42a05'
    # sequence_name = "0000cc6d8b108390"


    specs_path = os.path.join(RealEstate_path, sequence_name, f'{sequence_name}.txt')
    sequence_path = os.path.join(RealEstate_path, sequence_name, 'image_0')

    # Get a list of all poses [R,t] in this sequence
    poses = read_poses(specs_path)

    # Indices of 'good' image frames
    valid_indices = get_valid_indices(len(poses), sequence_path)
    
    # Get projection matrix from calib.txt, compute intrinsic K, and adjust K according to transformations
    original_image_size = torch.tensor(Image.open(os.path.join(sequence_path, f'{valid_indices[0]:06}.{IMAGE_TYPE}')).size)
    K = get_intrinsic_REALESTATE(specs_path, original_image_size)
    
    train_dataset = CustomDataset_first_two_thirds_train(sequence_path, poses, valid_indices, transform, K, dataset_type='train')
    val_dataset = CustomDataset_first_two_thirds_train(sequence_path, poses, valid_indices, transform, K, dataset_type='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=worker_init_fn)

    return train_loader, val_loader


def add_noise_to_F(F, noise_level):
    noise = torch.randn_like(F) * noise_level
    print(torch.mean(torch.abs(noise)))
    return F + noise


def vis():
    sequence_name = "0a610a129bdcc4e7"

    path = os.path.join("gt_epipolar_lines", f"{sequence_name}")
    good_frames_path = os.path.join(path, "good_frames")
    bad_frames_path = os.path.join(path, "bad_frames")
    if os.path.exists(good_frames_path) and os.path.exists(bad_frames_path):
        shutil.rmtree(good_frames_path)
        shutil.rmtree(bad_frames_path)

    train_loader, val_loader = data_with_one_sequence(batch_size=1, sequence_name=sequence_name)

    sed = 0
    for i,(first_image, second_image, label, unormalized_label,_) in enumerate(train_loader):
        first_image, second_image, label, unormalized_label = first_image.to(device), second_image.to(device), label.to(device), unormalized_label.to(device)

        epipolar_geo = EpipolarGeometry(first_image[0], second_image[0], F=label[0])
        sed += epipolar_geo.visualize(sqResultDir=path, file_num=i)

    print(sed/len(train_loader))




# if __name__ == "__main__":
    # vis()
        # train_loader, val_loader = data_with_one_sequence(batch_size=1, sequence_name="0000cc6d8b108390")
        # it = iter(train_loader)
        # first_image, second_image, label = next(it)
        # alpha, beta = find_coefficients(label)
        # f1 = label[0][:, 0]
        # f2 = label[0][:, 1]
        # f3 = label[0][:, 2]
        # print(torch.mean(alpha*f1 + beta*f2 - f3))
#
#         epoch_stats = {"algebraic_dist_truth": 0, "algebraic_dist_pred": 0, "algebraic_dist_pred_unormalized": 0,
#                                 "RE1_dist_truth": 0, "RE1_dist_pred": 0, "RE1_dist_pred_unormalized": 0,
#                                 "SED_dist_truth": 0, "SED_dist_pred": 0, "SED_dist_pred_unormalized": 0,
#                                 "avg_loss": 0, "avg_loss_R": 0, "avg_loss_t": 0, "epoch_penalty": 0, "file_num": 0}
#         sed1, sed2 = 0, 0
#         for i,(first_image, second_image, label, unormalized_label,_) in enumerate(val_loader):
#             first_image, second_image, label, unormalized_label = first_image.to(device), second_image.to(device), label.to(device), unormalized_label.to(device)
#
#             update_epoch_stats(epoch_stats, first_image[0], second_image[0], unormalized_label[0], label[0], label[0])
#
#
#
#
#         epoch_stats["algebraic_dist_truth"] = epoch_stats["algebraic_dist_truth"] / (i+1)
#         epoch_stats["algebraic_dist_pred"] = epoch_stats["algebraic_dist_pred"] / (i+1)
#         epoch_stats["algebraic_dist_pred_unormalized"] = epoch_stats["algebraic_dist_pred_unormalized"] / (i+1)
#         epoch_stats["RE1_dist_truth"] = epoch_stats["RE1_dist_truth"] / (i+1)
#         epoch_stats["RE1_dist_pred"] = epoch_stats["RE1_dist_pred"] / (i+1)
#         epoch_stats["RE1_dist_pred_unormalized"] = epoch_stats["RE1_dist_pred_unormalized"] / (i+1)
#         epoch_stats["SED_dist_truth"] = epoch_stats["SED_dist_truth"] / (i+1)
#         epoch_stats["SED_dist_pred"] = epoch_stats["SED_dist_pred"] / (i+1)
#         epoch_stats["SED_dist_pred_unormalized"] = epoch_stats["SED_dist_pred_unormalized"] / (i+1)
#
#         for key, value in epoch_stats.items():
#             # Convert tensor values to float for readability
#             if hasattr(value, 'item'):  # Checks if 'value' is a tensor
#                 value = value.item()  # Converts tensor to a Python number
#
#             print(f"{key}: {value}")
#         print("\n\n")

