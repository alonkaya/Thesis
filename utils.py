from params import *
from FunMatrix import read_poses
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

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

def plot_over_epoch(x, y, x_label, y_label, connecting_lines=True):
    
    if connecting_lines:
      plt.plot(x, y)
    else:
      plt.plot(x, y, marker='o', linestyle='')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{y_label} over {x_label}')
    plt.show()


def enforce_fundamental_constraints(F_vector):
    # Reshape the output vector into a 3x3 matrix
    F_matrix = F_vector.view(3, 3)

    # Use SVD to enforce the rank-2 constraint
    U, S, V = torch.svd(F_matrix)  

    S_prime = S.clone()  # Create a copy of S
    S_prime[-1] = 0  # Set the smallest singular value to zero
    
    F = U @ torch.diag(S) @ V.t() 

    # Normalize the matrix to ensure scale invariance
    # F = F_rank2 / torch.norm(F_rank2, p='fro')
    # F = F /  torch.max(torch.abs(F))

    return F


def generate_pose_and_frame_numbers(poses_path):
    poses = read_poses(poses_path)

    frame_numbers = [i for i in range(0, num_of_frames, jump_frames)]

    poses = [poses[i] for i in frame_numbers]

    return poses, frame_numbers