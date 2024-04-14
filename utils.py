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
    def __init__(self, in_channels, out_channels, kernel_size, groups):
        super(GroupedConvolution, self).__init__()
        self.grouped_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      groups=groups).to(device)
    
    def forward(self, x):
        return self.grouped_conv(x)
    

def plot(x, y1, y2, title, plots_path, x_label="Epochs", show=False, save=True):
    
    fig, axs = plt.subplots(1, 2, figsize=(18, 7))  # 1 row, 2 columns
    
    for ax, y_scale in zip(axs, ['linear', 'log']):
        ax.plot(x, y1, color='steelblue', label="Train")
        if y2 and len(y2)>0: ax.plot(x, y2, color='salmon', label="Test") 

        for i in range(0, len(y1), max(1, len(y1)//10)):
            ax.text(x[i], y1[i], f'{y1[i]:.4g}', fontsize=9, color='blue', ha='center', va='bottom')
            if y2: ax.text(x[i], y2[i], f'{y2[i]:.4g}', fontsize=9, color='red', ha='center', va='top')

        ax.set_xlabel(x_label)
        ax.set_ylabel(title if y_scale == 'linear' else f'{title} log scale')
        ax.set_title(f'{title} {y_scale} scale')
    
        ax.set_yscale(y_scale)
        ax.grid(True)
        ax.legend()


    if save:
        os.makedirs(plots_path, exist_ok=True)
        plt.savefig(f"""{plots_path}/{title}.png""")  # Specify the filename and extension

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

def norm_layer(unnormalized_x, predict_t=False, predict_pose=PREDICT_POSE):
    # Normalizes a batch of flattend 9-long vectors (i.e shape [-1, 9])
    if predict_pose and not predict_t:
        return normalize_L2(unnormalized_x)
    else:
        return normalize_L2(normalize_L1(unnormalized_x))
    

def check_nan(all_train_loss_last, all_val_loss_last, train_mae_last, val_mae_last, ec_err_pred_unoramlized_last, val_ec_err_pred_unormalized_last, ec_err_pred_last, all_penalty_last, plots_path):
    if math.isnan(all_train_loss_last) or math.isnan(all_val_loss_last) or math.isnan(train_mae_last) or math.isnan(val_mae_last) or math.isnan(ec_err_pred_unoramlized_last) or math.isnan(val_ec_err_pred_unormalized_last) or math.isnan(ec_err_pred_last) or math.isnan(all_penalty_last):
        print_and_write("found nan\n", plots_path)                
        return True
    return False
                     
def print_and_write(output, plots_path):
    os.makedirs(plots_path, exist_ok=True)
    output_path = os.path.join(plots_path, "output.log")
    with open(output_path, "a") as f:
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
    
    return (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

def init_main():
    faulthandler.enable()
    
    """When anomaly detection is enabled, PyTorch will perform additional checks during the backward pass
     to help locate the exact operation where an "anomalous" gradient (e.g., NaN or infinity) was produced. 
     But this jutrs perfomrance"""
    # torch.autograd.set_detect_anomaly(True)
    
    # Set up custom warning handling
    warnings.filterwarnings('always', category=RuntimeWarning)

    # Optionally, set NumPy error handling to 'warn' to catch overflow errors
    np.seterr(over='warn')

def find_coefficients(F):
    # Assuming F is a PyTorch tensor of shape (3, 3)
    # Extract columns f1, f2, and f3 from F
    f1 = F[:, 0:1]  # Column vector
    f2 = F[:, 1:2]  # Column vector
    f3 = F[:, 2]    # Vector

    # Stack f1 and f2 horizontally to form a 3x2 matrix
    A = torch.hstack([f1, f2])

    # Solve for alpha and beta using the least squares method
    # lstsq returns a named tuple, where the solution is the first item
    result  = torch.linalg.lstsq(A, f3)

    # Extract alpha and beta from the solution
    alpha = result.solution[0].item()
    beta = result.solution[1].item()

    return alpha, beta

def divide_by_dataloader(epoch_stats, len_train_loader, len_val_loader):
    for key, value in epoch_stats.items():
        if key.startswith("val_"):
            # For keys that start with "val_", divide by the length of val_loader
            epoch_stats[key] = value.cpu().item() / len_val_loader
        else:
            # For all other keys, divide by the length of train_loader
            # Assuming that 'file_num' should not be processed, we'll skip it
            if key != "file_num":
                epoch_stats[key] = value.cpu().item() / len_train_loader