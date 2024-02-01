import torch.optim as optim
import torch.nn as nn
from transformers import ViTModel, CLIPImageProcessor, CLIPVisionModel
from sklearn.metrics import mean_absolute_error
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from scipy.linalg import rq
import torch
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as T
import torch.multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1  
penalty_coeff = 2
jump_frames = 2
num_epochs = 50
learning_rate = 1e-4
mlp_hidden_sizes = [512, 256]
num_output = 9
clip_model_name = "openai/clip-vit-base-patch32"




class FMatrixRegressor(nn.Module):
    def __init__(self, mlp_hidden_sizes, num_output, pretrained_model_name, lr, freeze_pretrained_model=True):
        """
        Initialize the ViTMLPRegressor model.

        Args:
        - mlp_hidden_sizes (list): List of hidden layer sizes for the MLP.
        - num_output (int): Number of output units in the final layer.
        - pretrained_model_name (str): Name of the pretrained model to use.
        - lr (float): Learning rate for the optimizer.
        - device (str): Device to which the model should be moved (e.g., "cuda" or "cpu").
        - regress (bool): If True, use Mean Squared Error loss; if False, use Cross Entropy Loss.
        - freeze_pretrained_model (bool): If True, freeze the parameters of the pretrained model.
        """

        super(FMatrixRegressor, self).__init__()
        self.to(device)

        self.clip = True

        # Initialize CLIP processor and pretrained model
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(
            pretrained_model_name)
        self.pretrained_model = CLIPVisionModel.from_pretrained(
            pretrained_model_name).to(device)
        # self.pretrained_model = CLIPModel.from_pretrained(pretrained_model_name)

        # Get input dimension for the MLP based on CLIP configuration
        mlp_input_dim = self.pretrained_model.config.hidden_size
        # mlp_input_dim = self.pretrained_model.config.projection_dim


        # Freeze the parameters of the pretrained model if specified
        if freeze_pretrained_model:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        # Choose appropriate loss function based on regress parameter
        self.L2_loss = nn.MSELoss().to(device)
        self.L1_loss = nn.L1Loss().to(device)

        self.mlp = MLP(mlp_input_dim*7*7*2, mlp_hidden_sizes,
                       num_output).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x1, x2):
        x1 = self.clip_image_processor(images=x1, return_tensors="pt", do_resize=False, do_normalize=False,do_center_crop=False, do_rescale=False, do_convert_rgb=False).to(device)
        x2 = self.clip_image_processor(images=x2, return_tensors="pt", do_resize=False, do_normalize=False,do_center_crop=False, do_rescale=False, do_convert_rgb=False).to(device)

        x1_embeddings = self.pretrained_model(**x1).last_hidden_state[:, 1:, :].view(-1,  7*7*768).to(device)
        x2_embeddings = self.pretrained_model(**x2).last_hidden_state[:, 1:, :].view(-1,  7*7*768).to(device)

        # Concatenate both original and rotated embedding vectors
        embeddings = torch.cat([x1_embeddings, x2_embeddings], dim=1).to(device)

        # Train MLP on embedding vectors            
        unnormalized_output = self.mlp(embeddings).to(device).view(-1,3,3)

        output = normalize_L2(normalize_L1(output.view(-1,9))).view(-1,3,3)

        penalty = last_sing_value_penalty(output).to(device) 

        return unnormalized_output, output, penalty
    
    def train_model(self, train_loader, val_loader, num_epochs):
        # Lists to store training statistics
        all_train_loss, train_mae = [], []

        for epoch in range(num_epochs):
            self.train()
            labels, outputs = [], []
            avg_loss = 0

            for first_image, second_image, label, unormalized_label in train_loader:
                first_image, second_image, label, unormalized_label = first_image.to(device), second_image.to(device), label.to(device), unormalized_label.to(device)

                # Forward pass
                unnormalized_output, output, penalty = self.forward(first_image, second_image)

                # Compute loss
                l2_loss = self.L2_loss(output, label)
                loss = l2_loss + penalty_coeff *penalty 
                avg_loss += loss.detach()

                # Compute Backward pass and gradients
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Extend lists with batch statistics
                labels = labels.append(label)
                outputs = outputs.append(output)

            # Calculate and store mean absolute error for the epoch
            mae = torch.mean(torch.abs(torch.cat(labels, dim=0) - torch.cat(outputs, dim=0)))
            avg_loss /= len(train_loader)

            train_mae.append(mae.cpu())
            all_train_loss.append(avg_loss.cpu())    

            print(f"""Epoch {epoch+1}/{num_epochs}, Training Loss: {all_train_loss[-1]} Training MAE: {train_mae[-1]}""")


def normalize_L1(x):
    return x / torch.sum(torch.abs(x), dim=1, keepdim=True) 

def normalize_L2(x):
    return x / torch.linalg.norm(x, dim=1, keepdim=True)

def last_sing_value_penalty(output):
    # Compute the SVD of the output
    _, S, _ = torch.svd(output)

    # Add a term to the loss that penalizes the smallest singular value being far from zero
    rank_penalty = torch.mean(torch.abs(S[:, -1]))

    return rank_penalty

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


def get_intrinsic(calib_path):
    projection_matrix = read_calib(calib_path)

    # TODO: check if this func is correct
    # Step 1: Decompose the projection matrix P into the form P = K [R | t]
    # M = projection_matrix[:, :3]
    # K, _ = rq(M)
    # K = torch.tensor(K).to(device)

    # # Enforce positive diagonal for K
    # T = torch.diag(torch.sign(torch.diag(K)))
    # if torch.det(T) < 0:
    #     T[1, 1] *= -1

    # # Update K and R
    # K = torch.matmul(K.clone(), T)
    # # R = torch.matmul(T, R)

    # last_elem = K[2, 2]
    # K /= last_elem.clone()

    # return K
    M = projection_matrix[:, :3]
    K, R = rq(M)

    # Enforce positive diagonal for K
    T = np.diag(np.sign(np.diag(K)))
    if np.linalg.det(T) < 0:
        T[1, 1] *= -1

    # Update K and R
    K = np.dot(K, T)
    R = np.dot(T, R)

    K /= K[2, 2]

    return torch.tensor(K).to(device)


def adjust_intrinsic(k, original_size, resized_size, ceter_crop_size):
    # Adjust the intrinsic matrix K according to the transformations resize and center crop
    scale_factor = resized_size / original_size
    k[0, 0] *= scale_factor[0]  # fx
    k[1, 1] *= scale_factor[1]  # fy
    k[0, 2] *= scale_factor[0]  # cx
    k[1, 2] *= scale_factor[1]  # cy

    crop_offset = (resized_size - ceter_crop_size) / 2
    k[0, 2] -= crop_offset[0]  # cx
    k[1, 2] -= crop_offset[1]  # cy

    return k



def compute_relative_transformations(pose1, pose2):
    t1 = pose1[:, 3]
    R1 = pose1[:, :3]
    t2 = pose2[:, 3]
    R2 = pose2[:, :3]

    transposed_R1 = torch.transpose(R1, 0, 1)
    R_relative = torch.matmul(R2, transposed_R1)
    t_relative = torch.matmul(transposed_R1, (t2 - t1))
    # t_relative = t2 - np.dot(R_relative, t1)

    return R_relative, t_relative


def compute_essential(R, t):
    # Compute the skew-symmetric matrix of t
    t_x = torch.tensor([[0, -t[2], t[1]],
                        [t[2], 0, -t[0]],
                        [-t[1], t[0], 0]]).to(device)

    # Compute the essential matrix E
    E = torch.matmul(t_x, R)
    return E

def compute_fundamental(E, K1, K2):
    K2_inv_T = torch.transpose(torch.linalg.inv(K2), 0, 1)
    K1_inv = torch.linalg.inv(K1)

    # Compute the Fundamental matrix
    F = torch.matmul(K2_inv_T, torch.matmul(E, K1_inv))

    if torch.linalg.matrix_rank(F) != 2:
        print("rank of ground-truch not 2")

    return F

def get_F(poses, idx, K):
    R_relative, t_relative = compute_relative_transformations(poses[idx], poses[idx+jump_frames])
    E = compute_essential(R_relative, t_relative)
    F = compute_fundamental(E, K, K)

    return F


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_path, poses, transform, K):
        self.sequence_path = sequence_path
        self.sequence_num = sequence_path.split('/')[1]
        self.poses = poses
        self.transform = transform
        self.k = K
        self.valid_indices = self.get_valid_indices()

    def __len__(self):
        return len(self.poses) - jump_frames

    def __getitem__(self, idx):
        original_first_image = Image.open(os.path.join(self.sequence_path, f'{idx:06}.png'))
        original_second_image = Image.open(os.path.join(self.sequence_path, f'{idx+jump_frames:06}.png'))
        # if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        #     return None  # Return None if images don't exist

        # Transform: Resize, center, grayscale
        first_image = self.transform(original_first_image).to(device)
        second_image = self.transform(original_second_image).to(device)

        # Adjust K according to resize and center crop transforms and compute ground-truth F matrix
        adjusted_K = adjust_intrinsic(self.k.clone(), torch.tensor(original_first_image.size).to(device), torch.tensor([256, 256]).to(device), torch.tensor([224, 224]).to(device))
        
        unnormalized_F = get_F(self.poses, idx, adjusted_K)

        # Normalize F-Matrix
        F = normalize_L2(normalize_L1(unnormalized_F.view(-1,9))).view(-1,3,3)
        return first_image, second_image, F, unnormalized_F
    
def get_data_loaders():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),  # Converts to tensor and rescales [0,255] -> [0,1]
        # TODO: Normalize images?
    ])    
    
    sequence_paths1 = [f'sequences/00/image_0']
    poses_paths1 = [f'poses/00.txt']
    calib_paths1 = [f'sequences/00/calib.txt']

    sequence_paths2 = [f'sequences/01/image_0']
    poses_paths2 = [f'poses/01.txt']
    calib_paths2 = [f'sequences/01/calib.txt']    

    poses1 = read_poses(poses_paths1)
    poses2 = read_poses(poses_paths2)
    
    K1 = get_intrinsic(calib_paths1)
    K2 = get_intrinsic(calib_paths2)

    train_dataset = CustomDataset(sequence_paths1, poses1, transform, K1)     
    val_dataset = CustomDataset(sequence_paths2, poses2, transform, K2)

    train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)    

    return train_loader, val_loader


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    model = FMatrixRegressor(mlp_hidden_sizes, num_output, pretrained_model_name=clip_model_name,lr=learning_rate, freeze_pretrained_model=False).to(device)


    train_loader, val_loader = get_data_loaders()

    print(f'learning_rate: {learning_rate}, mlp_hidden_sizes: {mlp_hidden_sizes}, jump_frames: {jump_frames}, batch_size: {batch_size}')

    model.train_model(train_loader, val_loader, num_epochs=num_epochs)
