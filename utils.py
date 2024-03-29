from params import *
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import math
import numpy as np
import warnings
import torch.multiprocessing as mp
import os
import faulthandler

class MLP(nn.Module):
    def __init__(self, num_input, mlp_hidden_sizes, num_output, batchnorm_and_dropout):
        super(MLP, self).__init__()
        mlp_layers = []
        prev_size = num_input
        for hidden_size in mlp_hidden_sizes:
            mlp_layers.append(nn.Linear(prev_size, hidden_size))
            if batchnorm_and_dropout:
                mlp_layers.append(nn.BatchNorm1d(hidden_size))  # Batch Normalization
            mlp_layers.append(nn.ReLU())
            if batchnorm_and_dropout:
                mlp_layers.append(nn.Dropout())  # Dropout
            prev_size = hidden_size
        mlp_layers.append(nn.Linear(prev_size, num_output))

        self.layers = nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.layers(x)

class GroupedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, groups):
        super(GroupedConvolution, self).__init__()
        self.grouped_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      padding=padding,
                                      groups=groups).to(device)
    
    def forward(self, x):
        return self.grouped_conv(x)
    

def plot_over_epoch(x, y1, y2, title, penalty_coeff, batch_size, batchnorm_and_dropout, lr_mlp, lr_vit, x_label="Epochs", show=False, save=True, overfitting=False, average_embeddings=False, model=CLIP_MODEL_NAME, augmentation=AUGMENTATION, enforce_rank_2=ENFORCE_RANK_2, predict_pose=PREDICT_POSE, use_reconstruction=USE_RECONSTRUCTION_LAYER, RE1_coeff=RE1_COEFF):
    model_name = "CLIP" if model == CLIP_MODEL_NAME else "Google ViT"
    
    fig, axs = plt.subplots(1, 2, figsize=(18, 7))  # 1 row, 2 columns
    
    for ax, y_scale in zip(axs, ['linear', 'log']):
        try:
            ax.plot(x, y1, color='steelblue', label="Train")
            if y2 and len(y2)>0: ax.plot(x, y2, color='salmon', label="Test") 

            for i in range(0, len(y1), max(1, len(y1)//10)):
                ax.text(x[i], y1[i], f'{y1[i]:.4f}', fontsize=9, color='blue', ha='center', va='bottom')
                if y2: ax.text(x[i], y2[i], f'{y2[i]:.4f}', fontsize=9, color='red', ha='center', va='top')

            ax.set_xlabel(x_label)
            ax.set_ylabel(title if y_scale == 'linear' else f'{title} log scale')
            ax.set_title(f'{y_scale} scale')
        
            ax.set_yscale(y_scale)
            ax.grid(True)
            ax.legend()
        except Exception as e:
            print_and_write(e)

    if save:
        directory_name = f"""SVD_coeff {penalty_coeff} RE1_coeff {RE1_coeff} lr {lr_mlp} avg embeddings {average_embeddings} model {model_name} augmentation {AUGMENTATION} Force_rank_2 {enforce_rank_2} predict_pose {predict_pose} use_reconstruction {use_reconstruction}"""
        dir_path = os.path.join('plots', 'only_one_sequence', directory_name)
        os.makedirs(dir_path, exist_ok=True)

        plt.savefig(f"""{dir_path}/{title}.png""")  # Specify the filename and extension

    if show:
        plt.show()


def read_camera_intrinsic(path_to_intrinsic):
     with open(path_to_intrinsic, 'r') as f:
        lines = f.readlines()  # Read all lines into a list

        intrinsic_strings = lines[1].split()[1:5] if USE_REALESTATE else lines[0].split()[1:]

        return torch.tensor([float(x) for x in intrinsic_strings])

# Define a function to read the pose files in the poses folder
def read_poses(poses_path):
    poses = []
    with open(poses_path, 'r') as f:
        for i, line in enumerate(f):
            if USE_REALESTATE and i == 0: continue

            line = torch.tensor([float(x) for x in line.strip().split()])
            pose = line[7:] if USE_REALESTATE else line
            poses.append(pose.view(3, 4))

    return torch.stack(poses)

def normalize_max(x):
    return x / (torch.max(torch.abs(x), dim=1, keepdim=True)[0] + 1e-8)

def normalize_L1(x):
    return x / torch.sum(torch.abs(x), dim=1, keepdim=True) 

def normalize_L2(x):
    return x / torch.linalg.norm(x, dim=1, keepdim=True)


def norm_layer(unnormalized_x, predict_t=False, predict_pose=PREDICT_POSE, use_reconstruction=USE_RECONSTRUCTION_LAYER):
    # Normalizes a batch of flattend 9-long vectors (i.e shape [-1, 9])
    if use_reconstruction or predict_pose and predict_t:
        return normalize_max(unnormalized_x)
    
    elif predict_pose and not predict_t:
        return normalize_L2(unnormalized_x)
    
    else:
        return normalize_L2(normalize_L1(unnormalized_x))
    

def check_nan(all_train_loss_last, all_val_loss_last, train_mae_last, val_mae_last, ec_err_pred_unoramlized_last, val_ec_err_pred_unormalized_last, ec_err_pred_last, all_penalty_last):
    if math.isnan(all_train_loss_last) or math.isnan(all_val_loss_last) or math.isnan(train_mae_last) or math.isnan(val_mae_last) or math.isnan(ec_err_pred_unoramlized_last) or math.isnan(val_ec_err_pred_unormalized_last) or math.isnan(ec_err_pred_last) or math.isnan(all_penalty_last):
        print_and_write("found nan\n")                
        return True
    return False
                     
def print_and_write(output):
    with open("output.txt", "a") as f:
        f.write(output)
        print(output)

def not_learning(all_train_loss, all_val_loss):
    return len(all_train_loss) > 3 and abs(all_train_loss[-1] - all_train_loss[-2]) < 1e-4 and abs(all_train_loss[-1] - all_train_loss[-3]) < 1e-4  and abs(all_train_loss[-1] - all_train_loss[-4]) < 1e-4\
                                   and abs(all_val_loss[-1] - all_val_loss[-2]) < 1e-4 and abs(all_val_loss[-1] - all_val_loss[-3]) < 1e-4  and abs(all_val_loss[-1] - all_val_loss[-4]) < 1e-4

def reverse_transforms(img_tensor, mean=norm_mean, std=norm_std):
    """ Reverses the scaling and normalization transformation applied on the image.
        This function is called when computing the epipolar error.
    """
    # The mean and std have to be reshaped to [3, 1, 1] to match the tensor dimensions for broadcasting
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img_tensor = img_tensor * std + mean
    
    try:
        img = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    except Warning as e:
        print_and_write(f"warning: {e}, img_tensor: {img_tensor}")

    return img

def init_main():
    faulthandler.enable()

    if NUM_WORKERS > 0:
        mp.set_start_method('spawn', force=True)
        
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    torch.autograd.set_detect_anomaly(True)
    
    # Set up custom warning handling
    warnings.filterwarnings('always', category=RuntimeWarning)

    # Optionally, set NumPy error handling to 'warn' to catch overflow errors
    np.seterr(over='warn')

    print_and_write("###########################################################################################################\n\n")


def geodesic_error(R, R_star):
    # Compute the product of R transpose and R_star
    R_T_R_star = torch.matmul(R.transpose(-2, -1), R_star)

    # Compute the trace of the product
    trace = torch.diagonal(R_T_R_star, dim1=-2, dim2=-1).sum(-1)

    # Compute the geodesic error using the provided formula
    error = torch.acos(((trace - 1) / 2).clamp(-1, 1))  # Clamping for numerical stability

    return error

def get_embeddings(self, x1, x2, predict_t=False):
    if self.clip:  
        try:
            processor = self.clip_image_processor_t if predict_t else self.clip_image_processor
            model = self.pretrained_model_t if predict_t else self.pretrained_model

            x1 = processor(images=x1, return_tensors="pt", do_resize=False, do_normalize=False, do_center_crop=False, do_rescale=False, do_convert_rgb=False).to(device)
            x2 = processor(images=x2, return_tensors="pt", do_resize=False, do_normalize=False, do_center_crop=False, do_rescale=False, do_convert_rgb=False).to(device)

            x1_embeddings = model(**x1).last_hidden_state[:, 1:, :].view(-1, 7*7*self.model_hidden_size).to(device)
            x2_embeddings = model(**x2).last_hidden_state[:, 1:, :].view(-1, 7*7*self.model_hidden_size).to(device)    

        except Exception as e:
            print_and_write(f'clip: {e}')
    else:
        x1_embeddings = self.pretrained_model(x1).last_hidden_state[:, 1:, :].view(-1,  7*7*self.model_hidden_size).to(device)
        x2_embeddings = self.pretrained_model(x2).last_hidden_state[:, 1:, :].view(-1,  7*7*self.model_hidden_size).to(device)

    if self.average_embeddings:
        try:
            avg_patches = nn.AdaptiveAvgPool2d(1)
            x1_embeddings = avg_patches(x1_embeddings.view(-1, self.model_hidden_size, 7, 7)).view(-1, self.model_hidden_size)
            x2_embeddings = avg_patches(x2_embeddings.view(-1, self.model_hidden_size, 7, 7)).view(-1, self.model_hidden_size)
        except Exception as e: 
            print_and_write(f'avg_patches: {e}')

    if GROUP_CONV["use"]:
        grouped_conv_layer = GroupedConvolution(in_channels=self.model_hidden_size,   # Total input channels
                                out_channels=GROUP_CONV["out_channels"],  # Total output channels you want
                                kernel_size=3,
                                padding=1,
                                groups=GROUP_CONV["num_groups"])
        x1_embeddings = grouped_conv_layer(x1_embeddings.unsqueeze(2).unsqueeze(3)).view(-1, self.model_hidden_size//3)
        x2_embeddings = grouped_conv_layer(x2_embeddings.unsqueeze(2).unsqueeze(3)).view(-1, self.model_hidden_size//3)

    embeddings = torch.cat([x1_embeddings, x2_embeddings], dim=1)

    return embeddings