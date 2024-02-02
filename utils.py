from params import *
import matplotlib.pyplot as plt
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_input, mlp_hidden_sizes, num_output):
        super(MLP, self).__init__()
        mlp_layers = []
        prev_size = num_input
        for hidden_size in mlp_hidden_sizes:
            mlp_layers.append(nn.Linear(prev_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            prev_size = hidden_size
        mlp_layers.append(nn.Linear(prev_size, num_output))

        self.layers = nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.layers(x)


def plot_over_epoch(x, y, x_label, y_label, show=True, connecting_lines=True):
    plt.figure()
    if connecting_lines:
        plt.plot(x, y)
    else:
        plt.plot(x, y, marker='o', linestyle='')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{y_label} over {x_label}')

    plt.savefig(f'plots/{y_label}.png')  # Specify the filename and extension

    if show:
        plt.show()

# Define a function to read the calib.txt file
def read_calib(calib_path):
    with open(calib_path, 'r') as f:
        return torch.tensor([float(x) for x in f.readline().split()[1:]]).reshape(3, 4)


# Define a function to read the pose files in the poses folder
def read_poses(poses_path):
    poses = []
    with open(poses_path, 'r') as f:
        for line in f:
            pose = torch.tensor([float(x)
                                for x in line.strip().split()]).reshape(3, 4)
            poses.append(pose)

    return torch.stack(poses).to(device)

def normalize_max(x):
    return x / (torch.max(torch.abs(x), dim=1, keepdim=True)[0] + 1e-8)

def normalize_L1(x):
    return x / torch.sum(torch.abs(x), dim=1, keepdim=True) 

def normalize_L2(x):
    return x / torch.linalg.norm(x, dim=1, keepdim=True)


def norm_layer(unnormalized_x):
    # Normalizes a batch of flattend 9-long vectors (i.e shape [-1, 9])
    if use_reconstruction_layer:
        return normalize_max(unnormalized_x)

    else:
        return normalize_L2(normalize_L1(unnormalized_x))

