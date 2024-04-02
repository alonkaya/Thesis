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
            
        try:
            original_first_image = torchvision.io.read_image(os.path.join(self.sequence_path, f'{idx:06}.{IMAGE_TYPE}'))
            original_second_image = torchvision.io.read_image(os.path.join(self.sequence_path, f'{idx+JUMP_FRAMES:06}.{IMAGE_TYPE}'))
        except Exception as e:
            print_and_write(f"1\nError in sequence: {self.sequence_path}, idx: {idx}, dataset_type: {self.dataset_type} sequence num: {self.sequence_num}\nException: {e}")
            return
        
        try:
            first_image = self.transform(original_first_image)
            second_image = self.transform(original_second_image)
        except Exception as e:
            print_and_write(f"2\nError in sequence: {self.sequence_path}, idx: {idx}, dataset_type: {self.dataset_type} sequence num: {self.sequence_num}\nException: {e}")
            return
        
        try:
            if PREDICT_POSE:
                unormalized_R, unormalized_t = compute_relative_transformations(self.poses[idx],self. poses[idx+JUMP_FRAMES])
                unormalized_label = torch.cat((unormalized_R, unormalized_t.view(3,1)), dim=-1)
            else:
                unormalized_label = get_F(self.poses, idx, self.k)
        except Exception as e:
            print_and_write(f"4\nError in sequence: {self.sequence_path}, idx: {idx}, dataset_type: {self.dataset_type} sequence num: {self.sequence_num}\nException: {e}")
            return
        
        try:
            if PREDICT_POSE:
                R, t = norm_layer(unormalized_R.view(-1, 9)).view(3,3), norm_layer(unormalized_t.view(-1, 3), predict_t=True).view(3)

                label = torch.cat((R, t.view(3,1)), dim=-1)
            else:               
                label = norm_layer(unormalized_label.view(-1, 9)).view(3,3)
        except Exception as e:
            print_and_write("5\n {e}")
            return
            
        return first_image, second_image, label, unormalized_label, self.k


def get_valid_indices(sequence_len, sequence_path):
    valid_indices = []
    for idx in range(sequence_len - JUMP_FRAMES):
        img1_path = os.path.join(sequence_path, f'{idx:06}.{IMAGE_TYPE}')
        img2_path = os.path.join(sequence_path, f'{idx+JUMP_FRAMES:06}.{IMAGE_TYPE}')

        if os.path.exists(img1_path) and os.path.exists(img2_path):
            valid_indices.append(idx)

    return valid_indices

transform = v2.Compose([
    v2.Resize((256, 256)),
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


def data_with_one_sequence(batch_size, CustomDataset_type, sequence_name='0adea9da21629b61'):
    RealEstate_path = 'RealEstate10K/train_images'
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


# def add_noise_to_F(F, noise_level):
#     noise = torch.randn_like(F) * noise_level
#     print(torch.mean(torch.abs(noise)))
#     return F + noise

# def make_rank_2(F):
#         U1, S1, Vt1 = torch.linalg.svd(F, full_matrices=False)

#         S1[-1] = 0

#         output = torch.matmul(torch.matmul(U1, torch.diag_embed(S1)), Vt1)

#         if torch.linalg.matrix_rank(output) != 2:
#             print(f'rank of ground-truth not 2: {torch.linalg.matrix_rank(F)}')
#         return output

# if __name__ == "__main__":
#     sequence_name = ["0adea9da21629b61"]
#     for seq in sequence_name:
#         train_loader, val_loader = data_with_one_sequence(batch_size=1,CustomDataset_type=CUSTOMDATASET_TYPE, sequence_name=seq)
        
#         epoch_stats = {"algebraic_dist_truth": 0, "algebraic_dist_pred": 0, "algebraic_dist_pred_unormalized": 0, 
#                                 "RE1_dist_truth": 0, "RE1_dist_pred": 0, "RE1_dist_pred_unormalized": 0, 
#                                 "SED_dist_truth": 0, "SED_dist_pred": 0, "SED_dist_pred_unormalized": 0, 
#                                 "avg_loss": 0, "avg_loss_R": 0, "avg_loss_t": 0, "epoch_penalty": 0, "file_num": 0}
#         sed1, sed2 = 0, 0
#         for i,(first_image, second_image, label, unormalized_label,_) in enumerate(val_loader):
#             first_image, second_image, label, unormalized_label = first_image.to(device), second_image.to(device), label.to(device), unormalized_label.to(device)

#             update_dists(epoch_stats, first_image, second_image, unormalized_label, label, label)
            # break
            # for key, value in epoch_stats.items():
            #     # Convert tensor values to float for readability
            #     if hasattr(value, 'item'):  # Checks if 'value' is a tensor
            #         value = value.item()  # Converts tensor to a Python number

            #     print(f"{key}: {value/(i+1)}")
            # print("\n\n")
            # epipolar_geo_unormalized = EpipolarGeometry(first_image[0], second_image[0], F=unormalized_label[0])
            # sed1 += epipolar_geo_unormalized.get_SED_distance()
            # sed2 += epipolar_geo_unormalized.get_SED_distance2()

            # epipolar_geo = EpipolarGeometry(first_image[0], second_image[0], F=label)
            # epipolar_geo_unormalized.visualize(sqResultDir='unormalized', file_num=i)
            # epipolar_geo.visualize(sqResultDir='normalized', file_num=i)
                
            # batch_ep_err_unnormalized, batch_ep_err = batch_ep_err_unnormalized/len(first_image), batch_ep_err/len(first_image)
            # avg_ep_err_unnormalized, avg_ep_err = avg_ep_err_unnormalized + batch_ep_err_unnormalized, avg_ep_err + batch_ep_err
        # epoch_stats["algebraic_dist_truth"] = epoch_stats["algebraic_dist_truth"] / (i+1)
        # epoch_stats["algebraic_dist_pred"] = epoch_stats["algebraic_dist_pred"] / (i+1)
        # epoch_stats["algebraic_dist_pred_unormalized"] = epoch_stats["algebraic_dist_pred_unormalized"] / (i+1)
        # epoch_stats["RE1_dist_truth"] = epoch_stats["RE1_dist_truth"] / (i+1)
        # epoch_stats["RE1_dist_pred"] = epoch_stats["RE1_dist_pred"] / (i+1)
        # epoch_stats["RE1_dist_pred_unormalized"] = epoch_stats["RE1_dist_pred_unormalized"] / (i+1)
        # epoch_stats["SED_dist_truth"] = epoch_stats["SED_dist_truth"] / (i+1)
        # epoch_stats["SED_dist_pred"] = epoch_stats["SED_dist_pred"] / (i+1)
        # epoch_stats["SED_dist_pred_unormalized"] = epoch_stats["SED_dist_pred_unormalized"] / (i+1)
        # # sed1 /= len(val_loader)
        # # sed2 /= len(val_loader)
        # # print(sed1, sed2)
        # for key, value in epoch_stats.items():
        #     # Convert tensor values to float for readability
        #     if hasattr(value, 'item'):  # Checks if 'value' is a tensor
        #         value = value.item()  # Converts tensor to a Python number

        #     print(f"{key}: {value}")
        # print("\n\n")

