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

def add_last_sing_value_penalty(output, loss):
    # Compute the SVD of the output
    _, S, _ = torch.svd(output)
    
    # Add a term to the loss that penalizes the smallest singular value being far from zero
    rank_penalty = torch.mean(torch.abs(S[:,-1]))

    # TODO: add penatly for having less then 2 singular values
    if torch.any(S[:, 1] == 0):
        print("oops")

    loss = loss + rank_penalty

    return loss

def generate_pose_and_frame_numbers(poses_path):
    poses = read_poses(poses_path)

    frame_numbers = [i for i in range(num_of_frames)]

    poses = [poses[i] for i in frame_numbers]

    return poses, frame_numbers

def reconstruction_module(x, device):
        def get_rotation(rx, ry, rz):
            # normalize input?
            R_x = torch.tensor([
                [1.,    0.,             0.],
                [0.,    torch.cos(rx),    -torch.sin(rx)],
                [0.,    torch.sin(rx),     torch.cos(rx)]
            ]).to(device)
            R_y = torch.tensor([
                [torch.cos(ry),    0.,    -torch.sin(ry)],
                [0.,            1.,     0.],
                [torch.sin(ry),    0.,     torch.cos(ry)]
            ]).to(device)
            R_z = torch.tensor([
                [torch.cos(rz),    -torch.sin(rz),    0.],
                [torch.sin(rz),    torch.cos(rz),     0.],
                [0.,            0.,             1.]
            ]).to(device)
            R = torch.matmul(R_x, torch.matmul(R_y, R_z))
            return R

        def get_inv_intrinsic(f):
            return torch.tensor([
                [-1/(f+1e-8),   0.,             0.],
                [0.,            -1/(f+1e-8),    0.],
                [0.,            0.,             1.]
            ]).to(device)

        def get_translate(tx, ty, tz):
            return torch.tensor([
                [0.,  -tz, ty],
                [tz,  0,   -tx],
                [-ty, tx,  0]
            ]).to(device)

        # def get_linear_comb(f0, f1, f2, f3, f4, f5, cf1, cf2):
        #     return torch.tensor([
        #         [f0,            f1,            f2],
        #         [f3,            f4,            f5],
        #         [cf1*f0+cf2*f3, cf1*f1+cf2*f4, cf1*f2+cf2*f5]
        #     ])

        def get_fmat(x):
            # F = K2^(-T)*R*[t]x*K1^(-1)
            # Note: only need out-dim = 8  
            K1_inv = get_inv_intrinsic(x[0])
            K2_inv = get_inv_intrinsic(x[1]) #TODO: K2 should be -t not just -1..
            R  = get_rotation(x[2], x[3], x[4])
            T  = get_translate(x[5], x[6], x[7])
            F  = torch.matmul(K2_inv,
                    torch.matmul(R, torch.matmul(T, K1_inv))).requires_grad_()

            # to get the last row as linear combination of first two rows
            # new_F = get_linear_comb(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])
            # new_F = get_linear_comb(flat[0], flat[1], flat[2], flat[3], flat[4], flat[5], x[6], x[7])
            # flat = tf.reshape(new_F, [-1])
            return F

        out = get_fmat(x)

        return out

def normalize_F(x):
    return x / (torch.max(torch.abs(x)) + 1e-8)