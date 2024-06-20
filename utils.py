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
    def __init__(self, input_dim, hidden_dims=CONV_HIDDEN_DIM):
        super(ConvNet, self).__init__()
        layers = []
        prev_dim = input_dim
        self.hidden_dims = hidden_dims

        for hidden_dim in hidden_dims:
            layers.append(nn.Conv2d(prev_dim, hidden_dim, kernel_size=3, padding=1))
            if NORM:
                layers.append(nn.BatchNorm2d(hidden_dim)) if BATCH_SIZE > 1 else layers.append(nn.LayerNorm([hidden_dim, 7, 7]))
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
    x = x[5:]
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

def normalize_max(x):
    return x / (torch.max(torch.abs(x), dim=1, keepdim=True)[0] + 1e-8)

def normalize_L1(x):
    return x / torch.sum(torch.abs(x), dim=1, keepdim=True) 

def normalize_L2(x):
    return x / torch.linalg.norm(x, dim=1, keepdim=True)

def norm_layer(unnormalized_x):
    # Normalizes a batch of flattend 9-long vectors (i.e shape [-1, 9])
    return normalize_L2(unnormalized_x)
    

def check_nan(all_train_loss_last, all_val_loss_last, train_mae_last, val_mae_last, plots_path):
    if math.isnan(all_train_loss_last) or math.isnan(all_val_loss_last) or math.isnan(train_mae_last) or math.isnan(val_mae_last):
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
    return (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

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
