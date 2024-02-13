from params import *
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import math
import numpy as np

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


def plot_over_epoch(x, y1, y2, title, penalty_coeff, batch_size, batchnorm_and_dropout, lr_mlp, lr_vit, x_label="Epochs", show=False):

    fig, axs = plt.subplots(1, 2, figsize=(18, 7))  # 1 row, 2 columns
    
    for ax, y_scale in zip(axs, ['linear', 'log']):
        ax.plot(x, y1, color='blue', label="Train")
        ax.plot(x, y2, color='orange', label="Val")

        for i, txt in enumerate(y1):
            ax.text(x[i], y1[i], f'{txt:.3f}', fontsize=8, color='blue', ha='center', va='bottom')

        # Annotate each point on the Val line
        for i, txt in enumerate(y2):
            ax.text(x[i], y2[i], f'{txt:.3f}', fontsize=8, color='green', ha='center', va='top')

        ax.set_xlabel(x_label)
        ax.set_ylabel(title if y_scale == 'linear' else f'{title} log scale')
        ax.set_title(f'{title} -\n coeff: {penalty_coeff}, batch size: {batch_size}, lr_mlp: {lr_mlp}, lr_vit: {lr_vit}, scale: {y_scale}')
    
        ax.set_yscale(y_scale)
        ax.set_xticks(x)
        ax.grid(True)
        ax.legend()

    os.makedirs('plots', exist_ok=True)
    plt.savefig(f"""plots/{title}  coeff {penalty_coeff} batch_size {batch_size} bn_and_dropout {batchnorm_and_dropout} lr_mlp {lr_mlp} lr_vit {lr_vit} jump frames {JUMP_FRAMES} RealEstate {USE_REALESTATE}.png""")  # Specify the filename and extension
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


def norm_layer(unnormalized_x):
    # Normalizes a batch of flattend 9-long vectors (i.e shape [-1, 9])
    if USE_RECONSTRUCTION_LAYER:
        return normalize_max(unnormalized_x)
    else:
        return normalize_L2(normalize_L1(unnormalized_x))

def check_nan(all_train_loss_last, all_val_loss_last, train_mae_last, val_mae_last, ec_err_pred_unoramlized_last, val_ec_err_pred_unormalized_last, ec_err_pred_last, all_penalty_last):
    if math.isnan(all_train_loss_last) or math.isnan(all_val_loss_last) or math.isnan(train_mae_last) or math.isnan(val_mae_last) or math.isnan(ec_err_pred_unoramlized_last) or math.isnan(val_ec_err_pred_unormalized_last) or math.isnan(ec_err_pred_last) or math.isnan(all_penalty_last):
        with open("output.txt", "a") as f:
            f.write("found nan")
            print("found nan")                
        return True
    return False
                     
def print_and_write(output):
    with open("output.txt", "a") as f:
        f.write(output)
        print(output)

def not_learning(all_train_loss, all_val_loss):
    return len(all_train_loss) > 3 and abs(all_train_loss[-1] - all_train_loss[-2]) < 1e-3 and abs(all_train_loss[-1] - all_train_loss[-3]) < 1e-3  and abs(all_train_loss[-1] - all_train_loss[-4]) < 1e-3\
                                   and abs(all_val_loss[-1] - all_val_loss[-2]) < 1e-3 and abs(all_val_loss[-1] - all_val_loss[-3]) < 1e-3  and abs(all_val_loss[-1] - all_val_loss[-4]) < 1e-3  

def reverse_transforms(img_tensor, mean=norm_mean, std=norm_std):
    # The mean and std have to be reshaped to [3, 1, 1] to match the tensor dimensions for broadcasting
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img_tensor = img_tensor * std + mean

    return (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)