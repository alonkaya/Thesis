from params import *
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import warnings

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


def norm_layer(unnormalized_x, predict_t=False, use_reconstruction=USE_RECONSTRUCTION_LAYER):
    # Normalizes a batch of flattend 9-long vectors (i.e shape [-1, 9])
        return normalize_L2(normalize_L1(unnormalized_x))
    

                     
def print_and_write(output):
    with open("output.txt", "a") as f:
        f.write(output)
        print(output)

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
    # Optionally, set NumPy error handling to 'warn' to catch overflow errors
    np.seterr(over='warn')

    torch.autograd.set_detect_anomaly(True)

    # Set up custom warning handling
    warnings.filterwarnings('always', category=RuntimeWarning)

    print_and_write("###########################################################################################################\n\n")
