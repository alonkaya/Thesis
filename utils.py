import random
from params import *
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import math
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


def norm_layer(x):
    # Normalize the angle and shift values to [-1, 1]
    if SHIFT_RANGE == 0:
        return x / ANGLE_RANGE
    elif ANGLE_RANGE == 0:
        return x / SHIFT_RANGE
    else:
        angle, shift = x[:, 0] / ANGLE_RANGE, x[:, 1:] / SHIFT_RANGE
        return torch.cat((angle.unsqueeze(1), shift), dim=1)    
    

def check_nan(all_train_loss_last, all_val_loss_last, plots_path):
    if math.isnan(all_train_loss_last) or math.isnan(all_val_loss_last):
        print_and_write("found nan\n", plots_path)                
        return True
    return False


def print_and_write(output, plots_path):
    os.makedirs(plots_path, exist_ok=True)
    output_path = os.path.join(plots_path, "output.log")
    with open(output_path, "a") as f:
        f.write(output)
        print(output)


def init_main():
    faulthandler.enable()
    
    """When anomaly detection is enabled, PyTorch will perform additional checks during the backward pass
     to help locate the exact operation where an "anomalous" gradient (e.g., NaN or infinity) was produced. 
     But this jutrs perfomrance"""
    # torch.autograd.set_detect_anomaly(True)
    
    # Set up custom warning handling
    warnings.filterwarnings('always', category=RuntimeWarning)

    # Optionally, set NumPy error handling to 'warn' to catch overflow errors
    # np.seterr(over='warn')


def divide_by_dataloader(epoch_stats, len_train_loader=0, len_val_loader=0, len_test_loader=0):
    for key, value in epoch_stats.items():
        if key == "file_num" or value.shape != torch.Size([]): continue

        if key.startswith("val_"):
            epoch_stats[key] = value.cpu().item() / len_val_loader
        elif key.startswith("test_"):
            epoch_stats[key] = value.cpu().item() / len_test_loader
        else:
            epoch_stats[key] = value.cpu().item() / len_train_loader

def send_to_device(epoch_stats):
    for key, value in epoch_stats.items():
        if isinstance(value, torch.Tensor):
            epoch_stats[key] = value.to(device)    


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    random.seed(seed)

