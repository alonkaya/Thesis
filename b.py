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


learning_rate = 1e-4
mlp_hidden_sizes=[512,256]
num_epochs = 70
clip_model_name = "openai/clip-vit-base-patch32"
jump_frames = 2
num_output = 9 
penalty_coeff = 2
batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FMatrixRegressor(nn.Module):
    def __init__(self, mlp_hidden_sizes, num_output, pretrained_model_name, lr, device, freeze_pretrained_model=True):
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
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(pretrained_model_name)
        self.pretrained_model = CLIPVisionModel.from_pretrained(pretrained_model_name).to(device)

        # Get input dimension for the MLP based on CLIP configuration
        mlp_input_dim = self.pretrained_model.config.hidden_size


        # Freeze the parameters of the pretrained model if specified
        if freeze_pretrained_model:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        # Choose appropriate loss function based on regress parameter
        self.L2_loss = nn.MSELoss().to(device)
        self.L1_loss = nn.L1Loss().to(device)

        self.mlp = MLP(mlp_input_dim*7*7*2, mlp_hidden_sizes, num_output).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)



    def forward(self, x1, x2):
        x1 = self.clip_image_processor(images=x1, return_tensors="pt", do_resize=False, do_normalize=False, do_center_crop=False, do_rescale=False, do_convert_rgb=False).to(device)
        x2 = self.clip_image_processor(images=x2, return_tensors="pt", do_resize=False, do_normalize=False, do_center_crop=False, do_rescale=False, do_convert_rgb=False).to(device)

        x1_embeddings = self.pretrained_model(**x1).last_hidden_state[:,:49,:].view(-1, 7*7*768).to(device)
        x2_embeddings = self.pretrained_model(**x2).last_hidden_state[:,:49,:].view(-1, 7*7*768).to(device)

        # Concatenate both original and rotated embedding vectors
        embeddings = torch.cat([x1_embeddings, x2_embeddings], dim=1).to(device)

        # Train MLP on embedding vectors
        unnormalized_output = self.mlp(embeddings).view(-1,3,3).to(device)

        # Compute penalty for last singular value 
        penalty = last_sing_value_penalty(unnormalized_output).to(device)
    
        # Apply L2 norm on top of L1 norm 
        output = torch.stack([normalize_L2(normalize_L1(x)) for x in unnormalized_output]).to(device)
    
        return unnormalized_output, output, penalty
        
        
    def train_model(self, train_loader, val_loader, num_epochs):
        # Lists to store training statistics
        all_train_loss = []
        train_mae = []
        for epoch in range(num_epochs):
            self.train()

            # Lists to store per-batch statistics
            labels = []
            outputs = []
            avg_loss = 0
            for first_image, second_image, label, unormalized_label in train_loader:
                first_image, second_image, label = first_image.to(device), second_image.to(device), label.to(device) 
                           
                # Foward pass
                unnormalized_output, output, penalty = self.forward(first_image, second_image)
                
                # Compute loss
                l2_loss = self.L2_loss(output, label)
                loss = l2_loss + penalty
                avg_loss += loss.detach().cpu().item()

                # Compute Backward pass and gradients
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Extend lists with batch statistics
                labels.append(label.detach().cpu())
                outputs.append(output.detach().cpu())
                # cosine_similarities.extend(cosine_similarity.tolist())
      
            # Calculate and store root training loss for the epoch
            mae = torch.mean(torch.abs(torch.cat(labels, dim=0) - torch.cat(outputs, dim=0)))
            avg_loss /= len(train_loader)

            train_mae.append(mae)
            all_train_loss.append(avg_loss)
            
            epoch_output = f"""Epoch {epoch+1}/{num_epochs}, Training Loss: {all_train_loss[-1]} Training MAE: {train_mae[-1]}"""
            with open("output.txt", "a") as f:
                f.write(epoch_output)
                print(epoch_output)     
                f.close()       

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

def normalize_L1(x):
    return x / torch.sum(torch.abs(x))

def normalize_L2(x):
    return x / torch.linalg.matrix_norm(x)


def last_sing_value_penalty(output):
    _, S, _ = torch.svd(output)
    
    rank_penalty = torch.mean(torch.abs(S[:,-1]))

    # TODO: add penatly for having less then 2 singular values
    if torch.any(S[:, 1] == 0):
        print("oops")

    return rank_penalty

# Define a function to read the calib.txt file
def process_calib(calib_path):
    with open(calib_path, 'r') as f:
        p0_matrix = np.array([float(x) for x in f.readline().split()[1:]]).reshape(3, 4)

    return p0_matrix


# Define a function to read the pose files in the poses folder
def read_poses(poses_path):
    poses = []    
    with open(poses_path, 'r') as f:
        for line in f: 
            pose = np.array([float(x) for x in line.strip().split()]).reshape(3, 4)
            poses.append(pose)

    return np.stack(poses)


def compute_relative_transformations(pose1, pose2):
    t1 = pose1[:, 3]
    R1 = pose1[:, :3]
    t2 = pose2[:, 3]
    R2 = pose2[:, :3]    

    transposed_R1 = np.transpose(R1)
    R_relative = np.dot(R2, transposed_R1)
    t_relative = np.dot(transposed_R1, (t2 - t1))
    # t_relative = t2 - np.dot(R_relative, t1)

    return R_relative, t_relative


# Define a function to compute the essential matrix E from the relative pose matrix M
def compute_essential(R, t):
    # Compute the skew-symmetric matrix of t
    t_x = np.array([[0, -t[2], t[1]], 
                    [t[2], 0, -t[0]], 
                    [-t[1], t[0], 0]])

    # Compute the essential matrix E
    E = t_x @ R
    return E

# Define a function to compute the fundamental matrix F from the essential matrix E and the projection matrices P0 and P1
def compute_fundamental(E, K1, K2):
    K2_inv_T = np.linalg.inv(K2).T
    K1_inv = np.linalg.inv(K1)
    
    # Compute the Fundamental matrix 
    F = np.dot(K2_inv_T, np.dot(E, K1_inv))

    if not np.linalg.matrix_rank(F) == 2:
        print("rank of ground-truch not 2")

    return F

def get_internal_param_matrix(P):
    # Step 1: Decompose the projection matrix P into the form P = K [R | t]
    M = P[:, :3]
    K, R = rq(M)

    # Enforce positive diagonal for K
    T = np.diag(np.sign(np.diag(K)))
    if np.linalg.det(T) < 0:
        T[1, 1] *= -1

    # Update K and R
    K = np.dot(K, T)
    R = np.dot(T, R)

    K /= K[2, 2]

    return K, R




class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_path, poses, transform, K):
        self.sequence_path = sequence_path
        self.poses = poses
        self.transform = transform
        self.k = K

    def __len__(self):
        return len(self.poses) - jump_frames

    def __getitem__(self, idx):
        # Create PIL images
        first_image = Image.open(os.path.join(self.sequence_path, f'{idx:06}.png'))
        second_image = Image.open(os.path.join(self.sequence_path, f'{idx+jump_frames:06}.png'))

        # Transform: Resize, center, grayscale
        first_image = self.transform(first_image)
        second_image = self.transform(second_image)

        # Compute relative rotation and translation matrices
        R_relative, t_relative = compute_relative_transformations(self.poses[idx], self.poses[idx+jump_frames])

        # # Compute the essential matrix E
        E = compute_essential(R_relative, t_relative)

        # Compute the fundamental matrix F
        F = compute_fundamental(E, self.k, self.k)

        # Convert to tensor and rescale [0,255] -> [0,1]
        first_image, second_image, F, unnormalized_F  = T.to_tensor(first_image), T.to_tensor(second_image), normalize_L2(normalize_L1(torch.tensor(F, dtype=torch.float32))), torch.tensor(F, dtype=torch.float32)
        
        # TODO: Normalize images?
        return first_image, second_image, F, unnormalized_F

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
])

sequence_path = 'sequences/00/image_0'
poses_path = 'poses/00.txt'
calib_path = 'sequences/00/calib.txt'

poses = read_poses(poses_path)

# Read the calib.txt file to get the projection matricx and compute intrinsic K
projection_matrix = process_calib(calib_path)
K, _ = get_internal_param_matrix(projection_matrix)

# Split the dataset based on the calculated samples. Get first train_samples of data for training and the rest for validation
train_dataset = CustomDataset(sequence_path, poses, transform, K)

# Create a DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)
val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1)






if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FMatrixRegressor(mlp_hidden_sizes, num_output, pretrained_model_name=clip_model_name, lr=learning_rate, device=device, freeze_pretrained_model=False).to(device)

    print(f'learning_rate: {learning_rate}, mlp_hidden_sizes: {mlp_hidden_sizes}, jump_frames: {jump_frames}, batch size = {batch_size}')
    
    model.train_model(train_loader, val_loader, num_epochs=num_epochs)