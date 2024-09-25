import sys
from params import *
from PlotFromText import *
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import math
import numpy as np
import warnings
import os
import faulthandler
import traceback


class MLP(nn.Module):
    def __init__(self, input_dim, mlp_hidden_sizes=MLP_HIDDEN_DIM, num_output=NUM_OUTPUT):
        super(MLP, self).__init__()
        mlp_layers = []
        prev_size = input_dim
        for hidden_size in mlp_hidden_sizes:
            mlp_layers.append(nn.Linear(prev_size, hidden_size))
            mlp_layers.append(nn.ReLU())
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

class ConvNet(nn.Module):
    def __init__(self, input_dim, batch_size, hidden_dims=CONV_HIDDEN_DIM):
        super(ConvNet, self).__init__()
        layers = []
        prev_dim = input_dim
        self.hidden_dims = hidden_dims

        for hidden_dim in hidden_dims:
            layers.append(nn.Conv2d(prev_dim, hidden_dim, kernel_size=3, padding=1))
            if NORM:
                layers.append(nn.BatchNorm2d(hidden_dim)) if batch_size > 1 else layers.append(nn.LayerNorm([hidden_dim, 7, 7]))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.conv_layers(x) # shape: (batch_size, hidden_dims[-1], 7, 7)

        # Pooling
        pooled_features, indices = self.pool(x) # Output shape is (batch_size, hidden_dims[-1], 3, 3)
        
        # normalize the indices by dividing each index by the total number of elements in the pooled feature map (i.e. 3 * 3 = 9).
        indices = indices.float() / (pooled_features.shape[2] * pooled_features.shape[3])
        indices = indices.expand_as(pooled_features)
        
        pooled_features_with_position = torch.cat((pooled_features, indices), dim=1) # Output shape: (batch_size, 2 * hidden_dims[-1], 3, 3)

        x = self.flatten(pooled_features_with_position) # shape (batch_size, 2 * hidden_dims[-1] * 3 * 3)

        return x

def plot(x, y1, y2, title, plots_path, x_label="Epochs", show=False, save=True):
    # if len(y1) > 3 and (y1[0] > y1[3] + 2000 or y2[0] > y2[3] + 2000):
    y1 = y1[5:]
    y2 = y2[5:]
    x = list(x[5:])
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

        intrinsic_strings_cam0 = lines[1].split()[1:5] if USE_REALESTATE else lines[0].split()[1:]
        intrinsic_strings_cam1 = "" if USE_REALESTATE else lines[1].split()[1:]

        return torch.tensor([float(x) for x in intrinsic_strings_cam0]), torch.tensor([float(x) for x in intrinsic_strings_cam1])

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

def read_F_FM(F_path):
    files = os.listdir(F_path)
    files.sort(key=lambda f: int(f.split('.')[0]))

    Fs = torch.tensor([])
    for p in files:
        F = torch.tensor([])
        with open(os.path.join(F_path, p), 'r') as f:
            for line in f:
                l = torch.tensor([float(x) for x in line.strip().split(",")])
                F = torch.cat((F, l.view(1,3)), dim=0)

        Fs = torch.cat((Fs, F.view(1,3,3)), dim=0)

    return Fs

def normalize_max(x):
    return x / (torch.max(torch.abs(x), dim=1, keepdim=True)[0] + 1e-8)

def normalize_L1(x):
    return x / torch.sum(torch.abs(x), dim=1, keepdim=True) 

def normalize_L2(x):
    return x / torch.linalg.norm(x, dim=1, keepdim=True)

def normalize_L22(x):
    return x / torch.linalg.norm(x, 'fro', keepdim=True)

def norm_layer(unnormalized_x):
    # Normalizes a batch of flattend 9-long vectors (i.e shape [-1, 9])
    print(f'by row: {normalize_L2(unnormalized_x).cpu().numpy()}')
    print(f'fro: {normalize_L22(unnormalized_x).cpu().numpy()}')
    return normalize_L2(unnormalized_x)
    

def check_nan(all_train_loss_last, all_val_loss_last, train_mae_last, val_mae_last, plots_path):
    if math.isnan(all_train_loss_last) or math.isnan(all_val_loss_last) or math.isnan(train_mae_last) or math.isnan(val_mae_last):
        print_and_write("found nan\n", plots_path)                
        return True
    return False

def not_learning(val_RE1, plots_path):
    if sum(val_RE1[-100:])/100 > 200: 
        print_and_write("not learning\n", plots_path)
        return True
    return False

def not_decreasing(val_loss, num_epochs, plots_path):
    x = int(num_epochs/8)
    if sum(val_loss[-x:]) > sum(val_loss[-2*x:-x]) + 0.05*x:
        print_and_write("### Not decreasing ###", plots_path)
        return True
    
def ready_to_break(val_loss):
    if (val_loss[-1] < val_loss[-2]) and (val_loss[-1] < val_loss[-3]) and (val_loss[-1] < val_loss[-4]) and (val_loss[-1] < val_loss[-5]):
        return True

def print_and_write(output, plots_path):
    os.makedirs(plots_path, exist_ok=True)

    if plots_path == "plots/Stereo/Winners/SED_0.5__L2_1__huber_1__lr_0.0001__conv__CLIP__use_reconstruction_True/BS_8__ratio_0.1__mid__frozen_4":
        stack_trace = ''.join(traceback.format_stack())
        print(stack_trace)
    output_path = os.path.join(plots_path, "output.log")
    with open(output_path, "a") as f:
        f.write(output)
        print(output)
        sys.stdout.flush()

def reverse_transforms(img_tensor, mean=norm_mean, std=norm_std, is_scaled=True):
    """ Reverses the scaling and normalization transformation applied on the image.
        This function is called when computing the epipolar error.
    """
    # The mean and std have to be reshaped to [3, 1, 1] to match the tensor dimensions for broadcasting
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img_tensor = (img_tensor * std + mean) * 255 if is_scaled else img_tensor
    return (img_tensor.permute(1, 2, 0).numpy()).astype(np.uint8)

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
    # Assuming F is a PyTorch tensor of shape [batch_size, 3, 3]
    # Extract columns f1, f2, and f3 from F
    f1 = F[:, :, 0:1]  # Shape: [batch_size, 3, 1]
    f2 = F[:, :, 1:2]  # Shape: [batch_size, 3, 1]
    f3 = F[:, :, 2]    # Shape: [batch_size, 3]

    # Stack f1 and f2 horizontally to form a 3x2 matrix for each batch
    A = torch.cat([f1, f2], dim=2)  # Shape: [batch_size, 3, 2]

    # Solve for alpha and beta using the least squares method for each batch
    # lstsq returns a named tuple, where the solution is the first item
    result = torch.linalg.lstsq(A, f3.unsqueeze(-1))  # Ensure f3 is [batch_size, 3, 1] for broadcasting

    # Extract alpha and beta from the solution
    alpha = result.solution[:, 0, 0]  # Shape: [batch_size, 1]
    beta = result.solution[:, 1, 0]  # Shape: [batch_size, 1]

    return alpha, beta

def divide_by_dataloader(epoch_stats, len_train_loader=0, len_val_loader=0, len_test_loader=0):
    for key, value in epoch_stats.items():
        if key == "file_num" or value.shape != torch.Size([]): continue

        if key.startswith("val_"):
            epoch_stats[key] = value.cpu().item() / len_val_loader
        elif key.startswith("test_"):
            epoch_stats[key] = value.cpu().item() / len_test_loader
        else:
            epoch_stats[key] = value.cpu().item() / len_train_loader


def points_histogram(distances):

    # Define bins, ensuring they are monotonically increasing
    bin_edges = torch.tensor([0, 0.5, 1, 2, 3, 4, 6, 10, 20, 50, 100, 200, 300,400])

    # Calculate histogram counts
    counts, edges = np.histogram(distances, bins=bin_edges)

    # Calculate the uniform width for each bar
    uniform_width = 0.7  # Fixed width for all bars

    # Using integer locations for bin centers
    bin_centers = np.arange(len(counts))

    # Plotting the histogram
    plt.figure(figsize=(10, 5))
    plt.bar(bin_centers, counts, width=uniform_width, align='center', edgecolor='black', color='blue')

    # Customize x-ticks to show ranges and prevent overlap
    labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges)-1)]
    plt.xticks(ticks=bin_centers, labels=labels)

    # Adding labels and title
    plt.xlabel('SED Distance')
    plt.ylabel('Frequency')
    plt.title('Histogram of SED distances')

    # Show the plot
    plt.show()
    # plt.savefig(f"""{plots_path}/histogram.png""")


def trim(data, precent):
    # Calculate the 95th percentile threshold
    threshold = torch.quantile(data, 1-precent)

    # Filter the data to keep only values below the threshold
    return data[data < threshold]

def send_to_device(epoch_stats):
    for key, value in epoch_stats.items():
        if isinstance(value, torch.Tensor):
            epoch_stats[key] = value.to(device)    

def avg_results(output_path):
    epochs, training_losses, val_losses, training_maes, val_maes, alg_dists, val_alg_dists, re1_dists, val_re1_dists, sed_dists, val_sed_dists, alg_sqr_dists, val_alg_sqr_dists = process_epoch_stats(output_path)
    epochs, training_losses, val_losses, training_maes, val_maes, alg_dists, val_alg_dists, re1_dists, val_re1_dists, sed_dists, val_sed_dists, alg_sqr_dists, val_alg_sqr_dists =  \
    epochs[-10:], training_losses[-10:], val_losses[-10:], training_maes[-10:], val_maes[-10:], alg_dists[-10:], val_alg_dists[-10:], re1_dists[-10:], val_re1_dists[-10:], sed_dists[-10:], val_sed_dists[-10:], alg_sqr_dists[-10:], val_alg_sqr_dists[-10:]
    print("Training loss: ", sum(training_losses)/len(training_losses))
    print("Validation loss: ", sum(val_losses)/len(val_losses))
    print("Training MAE: ", sum(training_maes)/len(training_maes))
    print("Validation MAE: ", sum(val_maes)/len(val_maes))
    print("Training Alg dist: ", sum(alg_dists)/len(alg_dists))
    print("Validation Alg dist: ", sum(val_alg_dists)/len(val_alg_dists))
    print("Training RE1 dist: ", sum(re1_dists)/len(re1_dists))
    print("Validation RE1 dist: ", sum(val_re1_dists)/len(val_re1_dists))
    print("Training SED dist: ", sum(sed_dists)/len(sed_dists))
    print("Validation SED dist: ", sum(val_sed_dists)/len(val_sed_dists))
    print("Training Alg sqr dist: ", sum(alg_sqr_dists)/len(alg_sqr_dists))
    print("Validation Alg sqr dist: ", sum(val_alg_sqr_dists)/len(val_alg_sqr_dists))



def adjust_points(keypoints_dict, idx, top_crop, left_crop, height, width):
    " Keypoints_dict: A dictionary containing the keypoints for each image e.g {0: (pts1, pts2), 1: (pts1, pts2), ...} "
    # Convert keypoints to torch tensors
    pts1 = torch.tensor(keypoints_dict[idx][0], dtype=torch.float32).to(device) # shape [num_keypoints, 3]
    pts2 = torch.tensor(keypoints_dict[idx][1], dtype=torch.float32).to(device) # shape [num_keypoints, 3]

    # Adjust keypoints for the resized image
    scale = torch.tensor([RESIZE / width, RESIZE / height, 1], dtype=torch.float32).unsqueeze(0).to(device) # shape [1, 2]
    pts1 *= scale
    pts2 *= scale

    # Filter and adjust keypoints for the cropped image
    mask = (pts1[:, 0] >= left_crop+1) & (pts1[:, 0] < left_crop + CROP-1) & (pts1[:, 1] >= top_crop+1) & (pts1[:, 1] < top_crop + CROP-1) & \
           (pts2[:, 0] >= left_crop+1) & (pts2[:, 0] < left_crop + CROP-1) & (pts2[:, 1] >= top_crop+1) & (pts2[:, 1] < top_crop + CROP-1) # shape [num_keypoints]
    pts1 = pts1[mask] 
    pts2 = pts2[mask]
    
    crop_offset = torch.tensor([left_crop, top_crop, 0], dtype=torch.float32).unsqueeze(0).to(device) # shape [1, 2]
    pts1 -= crop_offset
    pts2 -= crop_offset

    return pts1, pts2

def adjust_points_no_dict(pts1, pts2, top_crop, left_crop, height, width):
    # Adjust keypoints for the resized image
    scale = torch.tensor([RESIZE / width, RESIZE / height, 1], dtype=torch.float32).unsqueeze(0).to(device) # shape [1, 2]
    pts1 *= scale
    pts2 *= scale

    # Filter and adjust keypoints for the cropped image
    mask = (pts1[:, 0] >= left_crop+1) & (pts1[:, 0] < left_crop + CROP-1) & (pts1[:, 1] >= top_crop+1) & (pts1[:, 1] < top_crop + CROP-1) & \
           (pts2[:, 0] >= left_crop+1) & (pts2[:, 0] < left_crop + CROP-1) & (pts2[:, 1] >= top_crop+1) & (pts2[:, 1] < top_crop + CROP-1) # shape [num_keypoints]
    pts1 = pts1[mask] 
    pts2 = pts2[mask]
    
    crop_offset = torch.tensor([left_crop, top_crop, 0], dtype=torch.float32).unsqueeze(0).to(device) # shape [1, 2]
    pts1 -= crop_offset
    pts2 -= crop_offset

    return pts1, pts2

 
def load_keypoints(keypoints_path):
    "load keypoints in the format {idx: (pts1, pts2), ...}"
    keypoints_dict = {}
    with open(keypoints_path, 'r') as f:
        for line in f:
            parts = line.strip().split(';')
            idx = int(parts[0])
            pts1 = eval(parts[1])
            pts2 = eval(parts[2])
            keypoints_dict[idx] = (pts1, pts2)
    return keypoints_dict


def rename_files(folder_path):
    ### os.listdir() does not guarantee the order of the files as they appear in the folder!!!!!!!!!!
    for filename in os.listdir(folder_path):
        if not filename.startswith("SED_"):
            continue  # Skip files that do not match the expected format
        
        # Extract parts of the filename
        parts = filename.split("__")
        
        # Dictionary to hold the parts
        parts_dict = {}
        for part in parts:
            key_value = part.split("_", 1)
            if len(key_value) == 2:
                parts_dict[key_value[0]] = key_value[1]
        
        # Create the new filename based on the given conditions
        if parts_dict.get("conv") == "True":
            new_parts = [
                "SED_" + parts_dict["SED"],
                "auged_" + parts_dict["auged"],
                "L12_coeffs_" + parts_dict["L12_coeffs"],
                "lr_" + parts_dict["lr"],
                "conv_True",
                "model_" + parts_dict["model"],
                "use_reconstruction_" + parts_dict["use_reconstruction"],
                "BS_" + parts_dict["BS"],
                "WD_" + parts_dict["WD"]
            ]
        else:
            new_parts = [
                "SED_" + parts_dict["SED"],
                "auged_" + parts_dict["auged"],
                "L12_coeffs_" + parts_dict["L12_coeffs"],
                "lr_" + parts_dict["lr"],
                "avg_embeddings_" + parts_dict["avg_embeddings"],
                "model_" + parts_dict["model"],
                "use_reconstruction_" + parts_dict["use_reconstruction"],
                "BS_" + parts_dict["BS"],
                "WD_" + parts_dict["WD"]
            ]
        
        new_filename = "__".join(new_parts)
        
        # Construct full file paths
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{filename}' to '{new_filename}'")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
