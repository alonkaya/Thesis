from Dataset import get_data_loaders
from FMatrixRegressor import FMatrixRegressor
from FunMatrix import EpipolarGeometry
import numpy as np
import matplotlib.pyplot as plt
from params import *
import os
import torch

# Function to denormalize image
def denormalize(image, mean, std):
    image = image.clone().numpy().transpose((1, 2, 0))  # Change from (C, H, W) to (H, W, C)
    mean = np.array(mean)
    std = np.array(std)
    image = std * image + mean  # Denormalize
    image = np.clip(image, 0, 1)  # Clip to [0, 1] range
    return image

# Function to visualize a batch of images
def show_images(first_image, second_image):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # Denormalize and visualize the first image
    ax[0].imshow(denormalize(first_image[0], norm_mean, norm_std))
    ax[0].set_title('First Image')
    ax[0].axis('off')

    # Denormalize and visualize the second image
    ax[1].imshow(denormalize(second_image[0], norm_mean, norm_std))
    ax[1].set_title('Second Image')
    ax[1].axis('off')

    plt.show()


def move_bad_images():
    # change dataset returns 6 params instead of 4. comment unnecessary lines in visualize
    train_loader, val_loader = get_data_loaders(batch_size=1)
    try:
        for i, (img1, img2, label, idx, sequence_path) in enumerate(val_loader):
            sequence_path = os.path.split(sequence_path[0])[0]
            epipolar_geo = EpipolarGeometry(img1[0], img2[0], F=label[0])
            epipolar_geo.visualize(idx=idx.item(), sequence_path=sequence_path, move_bad_images=True)
    except Exception as e:
        valid_indices_of_dataset(val_loader, idx)
        print(e)
    try:
        for i, (img1, img2, label, idx, sequence_path) in enumerate(train_loader):
            sequence_path = os.path.split(sequence_path[0])[0]
            epipolar_geo = EpipolarGeometry(img1[0], img2[0], F=label[0])
            epipolar_geo.visualize(idx=idx.item(), sequence_path=sequence_path, move_bad_images=True)
    except Exception as e:
        valid_indices_of_dataset(train_loader, idx)
        print(e)

def valid_indices_of_dataset(train_loader, idx):
    # Check if the DataLoader's dataset is a ConcatDataset
    if isinstance(train_loader.dataset, torch.utils.data.ConcatDataset):
        dataset_list = train_loader.dataset.datasets
        # Determine which dataset the current batch is from based on the index
        # This requires understanding the structure of indices in ConcatDataset
        sample_dataset = None
        cumulative_length = 0
        for dataset in dataset_list:
            cumulative_length += len(dataset)
            if idx < cumulative_length:
                sample_dataset = dataset
                break
    else:
        # If it's not a ConcatDataset, it's straightforward
        sample_dataset = train_loader.dataset
    
    # Now print the valid_indices of the determined dataset
    if sample_dataset is not None:
        print("Valid indices of the current batch's dataset:", sample_dataset.valid_indices)
    else:
        print("Dataset not found for the current batch")

def vis_gt():
    train_loader, val_loader = get_data_loaders()
    total_sed = 0
    for i, (img1, img2, label, pts1, pts2, seq_name) in enumerate(train_loader):
        epipolar_geo = EpipolarGeometry(img1[0], img2[0], F=label[0], pts1=pts1[0], pts2=pts2[0])
        total_sed += epipolar_geo.visualize(idx=i, epipolar_lines_path=os.path.join("gt_epipole_lines_KITTI_cam0", seq_name[0]))
        if i == 300: break
    total_sed /= i
    print(f'SED distance: {total_sed}') 

def vis_trained(plots_path):
    model = FMatrixRegressor(lr_vit=2e-5, lr_mlp=2e-5, pretrained_path=plots_path)
    
    train_loader, val_loader = get_data_loaders(batch_size=1)
    for i, (img1, img2, label, seq_name) in enumerate(val_loader):
        img1, img2 = img1.to(device), img2.to(device)
        output, _, _, _ = model.forward(img1, img2)

        epipolar_geo = EpipolarGeometry(img1[0], img2[0], output[0].detach())
        epipolar_geo.visualize(idx=i, epipolar_lines_path=os.path.join("predicted_epipole_lines_2", seq_name[0]))


def sed_distance_gt():
    train_loader, val_loader = get_data_loaders(batch_size=1)
    total_sed = 0

    for i, (img1, img2, label, pts1, pts2, _) in enumerate(val_loader):
        img1, img2, label, pts1, pts2 = img1.to(device), img2.to(device), label.to(device), pts1.to(device), pts2.to(device)

        epipolar_geo_gt = EpipolarGeometry(img1[0], img2[0], label[0], pts1[0], pts2[0]) 
        total_sed += epipolar_geo_gt.get_mean_SED_distance()
        if i == 300: break

    total_sed /= i
    print(f'SED distance: {total_sed}') 

def sed_distance_trained(plots_path):
    model = FMatrixRegressor(lr_vit=2e-5, lr_mlp=2e-5, pretrained_path=plots_path)
    
    train_loader, val_loader = get_data_loaders(batch_size=1)
    total_sed = 0
    for i, (img1, img2, label) in enumerate(val_loader):
        img1, img2 = img1.to(device), img2.to(device)
        output, _, _, _ = model.forward(img1, img2)
        epipolar_geo = EpipolarGeometry(img1[0], img2[0], output[0].detach())
        total_sed += epipolar_geo.get_mean_SED_distance()
    
    total_sed /= i
    print(f'SED distance: {total_sed}')


def sed_histogram_trained(plots_path):
    model = FMatrixRegressor(lr_vit=2e-5, lr_mlp=2e-5, pretrained_path=plots_path)
    
    train_loader, val_loader = get_data_loaders(batch_size=1)
    for i, (img1, img2, label, seq_name) in enumerate(val_loader):
        img1, img2 = img1.to(device), img2.to(device)
        output, _, _, _ = model.forward(img1, img2)
        print(seq_name[0])
        epipolar_geo = EpipolarGeometry(img1[0], img2[0], output[0].detach())
        sed = epipolar_geo.get_mean_SED_distance(show_histogram=True, plots_path=plots_path)


if __name__ == "__main__":
    # plots_path = 'plots/RealEstate/SED_0.1__lr_2e-05__avg_embeddings_True__model_CLIP__use_reconstruction_True'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    sed_distance_gt()
