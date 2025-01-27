import time

import torchvision
from Dataset_FM import get_dataloader_FM
from utils import print_and_write, reverse_transforms
from FunMatrix import EpipolarGeometry, compute_fundamental, get_F, update_epoch_stats
from FMatrixRegressor import FMatrixRegressor
from Dataset import get_data_loaders
from params import *

import time
import cv2
import numpy as np
import os
import torch
import re

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
    train_loader, val_loader, test_loader = get_data_loaders(train_size=1, part='head', batch_size=1)

    for i, (img1, img2, label, pts1, pts2, seq_name) in enumerate(val_loader):
        print(pts1[0].shape)
        img1 = img1[0].cpu().detach()  # Shape (C, H, W)
        img2 = img2[0].cpu().detach()  # Shape (C, H, W)

        # Unnormalize the image
        img1_np = reverse_transforms(img1, norm_mean.cpu(), norm_std.cpu(), is_scaled=True)
        img2_np = reverse_transforms(img2, norm_mean.cpu(), norm_std.cpu(), is_scaled=True)

        # Get the first set of keypoints
        pts1_np = pts1[0].cpu().detach().numpy()  # Shape (N, 2)
        pts2_np = pts2[0].cpu().detach().numpy()  # Shape (N, 2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 8))  # 1 row, 2 columns
        # Plot the first image
        axes[0].imshow(img1_np)
        axes[0].scatter(pts1_np[:, 0], pts1_np[:, 1], c='red', s=10, marker='o')  # Plot keypoints on img1
        axes[0].set_title(f"Image 1 from sequence: {seq_name[0]}")
        axes[0].axis('off')

        # Plot the second image
        axes[1].imshow(img2_np)
        axes[1].scatter(pts2_np[:, 0], pts2_np[:, 1], c='blue', s=10, marker='o')  # Plot keypoints on img2
        axes[1].set_title(f"Image 2 from sequence: {seq_name[0]}")
        axes[1].axis('off')

        # Show the figure
        plt.tight_layout()
        plt.savefig("Flying")
        # time.sleep(2)
        break

        # if i==10: break
        

def vis_cognata():
    img0_path = "SceneFlow/Monkaa_cleanpass/flower_storm_x2/left/0000.png"
    img1_path = "SceneFlow/Monkaa_cleanpass/flower_storm_x2/right/0000.png"

    img0 = torchvision.io.read_image(img0_path)
    img1 = torchvision.io.read_image(img1_path)
    F_monkaa = torch.tensor([[ 0,     0,        0],
                             [ 0,     0,       -9.5238e-04],
                             [ 0,     9.5238e-04,     0]]).to(device)
    
    # R = torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype=torch.float32).to(device)
    # t = torch.tensor([-0.6, 0, 0], dtype=torch.float32).to(device)
    # E = torch.tensor([[0, -t[2], t[1]],
    #                     [t[2], 0, -t[0]],
    #                     [-t[1], t[0], 0]]).to(device)
    # k = torch.tensor([[  1372.4844291261174,     0.0,                640.0], 
    #                    [  0.0,                   1372.4844291261174, 200.0], 
    #                    [  0,                     0,                  1.0]])
    # F = compute_fundamental(E, k, k)
    # F = torch.tensor([[4.233936605179281233e-06,3.347417379233607752e-05,-1.758078923829230200e-02],
    # [-5.645440844391099376e-06,-4.503873870436473856e-05,2.423103527713659985e-02],
    # [-2.996264635733713070e-04,-2.243601003810080305e-03,1.000000000000000000e+00]])

    ep = EpipolarGeometry(img0, img1, F=F_monkaa.unsqueeze(0), is_scaled=False, threshold=0.3)
    pts0, pts1 = ep.pts1, ep.pts2

    # # Compute epipolar lines in img1 corresponding to points in img0
    # lines1 = cv2.computeCorrespondEpilines(pts0[:, :2].reshape(-1, 1, 2).numpy(), 1, F.numpy()).reshape(-1, 3)  # 1 indicates that points are in img0
    # img1_with_lines = ep.image2_numpy.copy()
    
    # thickness = 1
    # h, w = ep.image2_numpy.shape[:2]

    # # Draw each line in lines1 on img1
    # for line, pt in zip(lines1, pts1):
    #     color = tuple(np.random.randint(0, 255, 3).tolist())  # Random color for each line
    #     # line has the form [a, b, c] and represents the line equation ax + by + c = 0
    #     a, b, c = line
    #     # Calculate two points on the line to draw it within the image dimensions
    #     x0, y0 = 0, int(-c / b)  # When x = 0
    #     x1, y1 = w, int(-(c + a * w) / b)  # When x = width of the image

    #     # Draw the line on img1
    #     img1_with_lines = cv2.line(img1_with_lines, (x0, y0), (x1, y1), color, thickness)
    #     img1_with_lines = cv2.circle(img1_with_lines, (int(pt[0]), int(pt[1])), 2, color, -1)

    # # Show or save the image with epipolar lines
    # cv2.imshow("Epipolar Lines on img1", img1_with_lines)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    img0_pts = ep.image1_numpy.copy()
    img1_pts = ep.image2_numpy.copy()
    for i, (pt0, pt1) in enumerate(zip(pts0, pts1)):
        # if i == 30: break
        img0_pts = cv2.circle(img0_pts, (int(pt0[0]), int(pt0[1])), 5, (20, 20, 160), -1)
        img1_pts = cv2.circle(img1_pts, (int(pt1[0]), int(pt1[1])), 5, (20, 20, 160), -1)

    # Create padding (e.g., 10-pixel wide, white vertical strip)
    padding = 255 * np.zeros((img0_pts.shape[0], 30, 3), dtype=np.uint8)  # 10-pixel wide white space

    # Combine the two images with padding in between
    combined_image = np.hstack((img0_pts, padding, img1_pts))

    cv2.imshow("Epipolar Lines on img1", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def vis_trained():
    pretrained_path = "plots/Stereo/Winners/SED_0.5__L2_1__huber_1__lr_0.0001__conv__CLIP__use_reconstruction_True/BS_8__ratio_0.2__mid__frozen_0"
    model = FMatrixRegressor(lr=LR[0], batch_size=1, L2_coeff=1, huber_coeff=1, pretrained_path=pretrained_path)
    train_loader, val_loader, test_loader = get_data_loaders(train_size=0.3, part='head', batch_size=1)

    epoch_stats = {"algebraic_pred": torch.tensor(0), "algebraic_sqr_pred": torch.tensor(0), "RE1_pred": torch.tensor(0), "SED_pred": torch.tensor(0), 
                "val_algebraic_pred": torch.tensor(0), "val_algebraic_sqr_pred": torch.tensor(0), "val_RE1_pred": torch.tensor(0), "val_SED_pred": torch.tensor(0), 
                "test_algebraic_pred": torch.tensor(0), "test_algebraic_sqr_pred": torch.tensor(0), "test_RE1_pred": torch.tensor(0), "test_SED_pred": torch.tensor(0),
                "algebraic_truth": torch.tensor(0), "algebraic_sqr_truth": torch.tensor(0), "RE1_truth": torch.tensor(0), "SED_truth": torch.tensor(0), 
                "val_algebraic_truth": torch.tensor(0), "val_algebraic_sqr_truth": torch.tensor(0), "val_RE1_truth": torch.tensor(0), "val_SED_truth": torch.tensor(0), 
                "test_algebraic_truth": torch.tensor(0), "test_algebraic_sqr_truth": torch.tensor(0), "test_RE1_truth": torch.tensor(0), "test_SED_truth": torch.tensor(0),
                "loss": torch.tensor(0), "val_loss": torch.tensor(0), "test_loss": torch.tensor(0), 
                "labels": torch.tensor([]), "outputs": torch.tensor([]), "val_labels": torch.tensor([]), "val_outputs": torch.tensor([]), "test_labels": torch.tensor([]), "test_outputs": torch.tensor([]),
                "file_num": 0}
    
    for i, (img1, img2, label, pts1, pts2, seq_name) in enumerate(test_loader):
        img1, img2, label, pts1, pts2 = img1.to(device), img2.to(device), label.to(device), pts1.to(device), pts2.to(device)

        # output = model.forward(img1, img2)

        img1 = img1[0].cpu().detach()  # Shape (C, H, W)
        img2 = img2[0].cpu().detach()  # Shape (C, H, W)

        # Unnormalize the image
        img1_np = reverse_transforms(img1, norm_mean.cpu(), norm_std.cpu(), is_scaled=True)
        img2_np = reverse_transforms(img2, norm_mean.cpu(), norm_std.cpu(), is_scaled=True)

        # Get the first set of keypoints
        pts1_np = pts1[0].cpu().detach().numpy()  # Shape (N, 2)
        pts2_np = pts2[0].cpu().detach().numpy()  # Shape (N, 2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 8))  # 1 row, 2 columns
        # Plot the first image
        axes[0].imshow(img1_np)
        axes[0].scatter(pts1_np[:, 0], pts1_np[:, 1], c='red', s=10, marker='o')  # Plot keypoints on img1
        axes[0].set_title(f"Image 1 from sequence: {seq_name[0]}")
        axes[0].axis('off')

        # Plot the second image
        axes[1].imshow(img2_np)
        axes[1].scatter(pts2_np[:, 0], pts2_np[:, 1], c='blue', s=10, marker='o')  # Plot keypoints on img2
        axes[1].set_title(f"Image 2 from sequence: {seq_name[0]}")
        axes[1].axis('off')

        # Show the figure
        plt.tight_layout()
        plt.savefig("monkaa")

        break

def sed_distance_trained():
    pretrained_path = "plots/Stereo/Winners/SED_0.5__L2_1__huber_1__lr_0.0001__conv__CLIP__use_reconstruction_True/BS_8__ratio_0.2__mid__frozen_0"
    model = FMatrixRegressor(lr=LR[0], batch_size=1, L2_coeff=1, huber_coeff=1, pretrained_path=pretrained_path)
    train_loader, val_loader, test_loader = get_data_loaders(train_size=0.3, batch_size=1)

    epoch_stats = {"algebraic_pred": torch.tensor(0), "algebraic_sqr_pred": torch.tensor(0), "RE1_pred": torch.tensor(0), "SED_pred": torch.tensor(0), 
                "val_algebraic_pred": torch.tensor(0), "val_algebraic_sqr_pred": torch.tensor(0), "val_RE1_pred": torch.tensor(0), "val_SED_pred": torch.tensor(0), 
                "test_algebraic_pred": torch.tensor(0), "test_algebraic_sqr_pred": torch.tensor(0), "test_RE1_pred": torch.tensor(0), "test_SED_pred": torch.tensor(0),
                "algebraic_truth": torch.tensor(0), "algebraic_sqr_truth": torch.tensor(0), "RE1_truth": torch.tensor(0), "SED_truth": torch.tensor(0), 
                "val_algebraic_truth": torch.tensor(0), "val_algebraic_sqr_truth": torch.tensor(0), "val_RE1_truth": torch.tensor(0), "val_SED_truth": torch.tensor(0), 
                "test_algebraic_truth": torch.tensor(0), "test_algebraic_sqr_truth": torch.tensor(0), "test_RE1_truth": torch.tensor(0), "test_SED_truth": torch.tensor(0),
                "loss": torch.tensor(0), "val_loss": torch.tensor(0), "test_loss": torch.tensor(0), 
                "labels": torch.tensor([]), "outputs": torch.tensor([]), "val_labels": torch.tensor([]), "val_outputs": torch.tensor([]), "test_labels": torch.tensor([]), "test_outputs": torch.tensor([]),
                "file_num": 0}
    
    for i, (img1, img2, label, pts1, pts2, seq_name) in enumerate(test_loader):
        img1, img2, label, pts1, pts2 = img1.to(device), img2.to(device), label.to(device), pts1.to(device), pts2.to(device)

        output = model.forward(img1, img2)

        update_epoch_stats(epoch_stats, img1.detach(), img2.detach(), label.detach(), output, pts1, pts2, data_type="test")
        if i == 10000: break
    

    print(f"""SED distance: {epoch_stats["test_SED_pred"]/(i+1)}
Algebraic distance: {epoch_stats["test_algebraic_pred"]/(i+1)}
RE1 distance: {epoch_stats["test_RE1_pred"]/(i+1)}

SED distance truth: {epoch_stats["test_SED_truth"]/(i+1)}
Algebraic distance truth: {epoch_stats["test_algebraic_truth"]/(i+1)}
RE1 distance truth: {epoch_stats["test_RE1_truth"]/(i+1)}""")


def sed_gt():
    train_loader, val_loader, test_loader = get_data_loaders(train_size=0.3, batch_size=1)

    epoch_stats = {"test_algebraic_pred": torch.tensor(0), "test_algebraic_sqr_pred": torch.tensor(0), "test_RE1_pred": torch.tensor(0), "test_SED_pred": torch.tensor(0),
                   "test_algebraic_truth": torch.tensor(0), "test_algebraic_sqr_truth": torch.tensor(0), "test_RE1_truth": torch.tensor(0), "test_SED_truth": torch.tensor(0),
                   "test_loss": torch.tensor(0), "test_labels": torch.tensor([]), "test_outputs": torch.tensor([])}
    
    for i, (img1, img2, label, pts1, pts2, seq_name) in enumerate(test_loader):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        update_epoch_stats(epoch_stats, img1.detach(), img2.detach(), label.detach(), label.detach(), pts1, pts2, seq_name, data_type="test")
        print(epoch_stats["test_SED_pred"])

        if i==10: break

    print(f"""SED distance: {epoch_stats["test_SED_pred"]/(i+1)}
Algebraic distance: {epoch_stats["test_algebraic_pred"]/(i+1)}
RE1 distance: {epoch_stats["test_RE1_pred"]/(i+1)}""")


def move_bad_frames_realestate():
    train_loader, val_loader, test_loader = get_data_loaders(train_size=100000000, batch_size=1)

    
    for img1, img2, label, pts1, pts2, seq_name, seq_path, idx in train_loader:
        if seq_path==None:
            continue
        if img1 == None:
            seq_path_parent = os.path.dirname(seq_path[0])
            source_path = os.path.join(seq_path[0], f'{idx[0]:06}.jpg')
            dest_path = os.path.join(seq_path_parent, "bad_frames", f'{idx[0]:06}.png')
            os.makedirs(os.path.join(seq_path_parent, "bad_frames"), exist_ok=True)
            print(f'from: {source_path}, to: {dest_path}')
            os.rename(source_path, dest_path)

    for img1, img2, label, pts1, pts2, seq_name, seq_path, idx in val_loader:
        if seq_path==None:
            continue
        if img1 == None:
            seq_path_parent = os.path.dirname(seq_path[0])
            source_path = os.path.join(seq_path[0], f'{idx[0]:06}.jpg')
            dest_path = os.path.join(seq_path_parent, "bad_frames", f'{idx[0]:06}.png')
            os.makedirs(os.path.join(seq_path_parent, "bad_frames"), exist_ok=True)
            print(f'from: {source_path}, to: {dest_path}')
            os.rename(source_path, dest_path)

    for img1, img2, label, pts1, pts2, seq_name, seq_path, idx in test_loader:
        if seq_path==None:
            continue
        if img1 == None:
            seq_path_parent = os.path.dirname(seq_path[0])
            source_path = os.path.join(seq_path[0], f'{idx[0]:06}.jpg')
            dest_path = os.path.join(seq_path_parent, "bad_frames", f'{idx[0]:06}.png')
            os.makedirs(os.path.join(seq_path_parent, "bad_frames"), exist_ok=True)
            print(f'from: {source_path}, to: {dest_path}')
            os.rename(source_path, dest_path)            


def return_bad_frames_to_seq():
    RealEstate_paths = ['RealEstate10K/train_images', 'RealEstate10K/val_images']

    for RealEstate_path in RealEstate_paths:
        for sequence_name in os.listdir(RealEstate_path): 
            bad_seq_path = os.path.join(RealEstate_path, sequence_name, 'bad_frames')
            image_0_path = os.path.join(RealEstate_path, sequence_name, 'image_0')
            if os.path.exists(bad_seq_path):
                for img in os.listdir(bad_seq_path):
                    print(f'from: {os.path.join(bad_seq_path, img)}, to: {os.path.join(image_0_path, img)}') 
                    # os.rename(os.path.join(bad_seq_path, img), os.path.join(image_0_path, img))

def sed_histogram_trained(plots_path):
    model = FMatrixRegressor(lr_vit=2e-5, lr_mlp=2e-5, pretrained_path=plots_path)
    
    train_loader, val_loader = get_data_loaders(batch_size=1)
    for i, (img1, img2, label, seq_name) in enumerate(val_loader):
        img1, img2 = img1.to(device), img2.to(device)
        output, _, _, _ = model.forward(img1, img2)
        print(seq_name[0])
        epipolar_geo = EpipolarGeometry(img1[0], img2[0], output[0].detach())
        sed = epipolar_geo.get_mean_SED_distance(show_histogram=True, plots_path=plots_path)

def sed_vs_rotation_translation(file_path):
    # Load the data
    with open(file_path, 'r') as file:
        data = file.read()

    # Regular expressions to match the necessary data
    idx_pattern = re.compile(r'idx:\s*(\d+)')
    sed_pattern = re.compile(r'SED:\s*([\d\.]+)')
    r_pattern = re.compile(r'R:\s*\[\[([-\d\.\se\s\[\]\n]+)\]\]')
    t_pattern = re.compile(r't:\s*\[([-\d\.\se\s]+)\]')

    # Extract the data
    indices = idx_pattern.findall(data)
    seds = sed_pattern.findall(data)
    rotations = r_pattern.findall(data)
    translations = t_pattern.findall(data)

    # Convert extracted data to appropriate types
    indices = list(map(int, indices))
    seds = list(map(float, seds))

    def clean_matrix_str(matrix_str):
        matrix_str = matrix_str.replace('\n', ' ').replace('] [', '];[').replace('[', '').replace(']', '')
        return matrix_str

    def parse_matrix(matrix_str):
        cleaned_str = clean_matrix_str(matrix_str)
        return np.array([list(map(float, row.split())) for row in cleaned_str.split(';')])

    rotations = [parse_matrix(rot) for rot in rotations]
    translations = [np.array(list(map(float, t.split()))) for t in translations]

    # Ensure all lists are of the same length
    lengths = [len(indices), len(seds), len(rotations), len(translations)]
    min_length = min(lengths)
    indices = indices[:min_length]
    seds = seds[:min_length]
    rotations = rotations[:min_length]
    translations = translations[:min_length]

    # Calculate the rotation angles (in degrees) from the rotation matrices using angle-axis representation
    def rotation_angle_from_matrix(matrix):
        angle = np.arccos((np.trace(matrix) - 1) / 2)
        if np.isnan(angle):
            angle = 0  # Handle numerical errors for very small rotations
        return np.degrees(angle)

    rotation_angles = [rotation_angle_from_matrix(r) for r in rotations]
    translation_magnitudes = [np.linalg.norm(t) for t in translations]

    # Calculate correlation coefficients
    rotation_angle_corr = np.corrcoef(rotation_angles, seds)[0, 1]
    translation_magnitude_corr = np.corrcoef(translation_magnitudes, seds)[0, 1]

    print("Correlation between SED and Rotation Angle:", rotation_angle_corr)
    print("Correlation between SED and Translation Magnitude:", translation_magnitude_corr)

    # Normalize rotation angles and translation magnitudes for better visualization
    normalized_rotation_angles = np.array(rotation_angles) - 90
    normalized_translation_magnitudes = np.array(translation_magnitudes) - np.mean(translation_magnitudes)

    # Plotting the normalized data
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(normalized_rotation_angles, seds, c='blue', label='SED vs Normalized Rotation Angle')
    plt.xlabel('Normalized Rotation Angle (degrees)')
    plt.ylabel('SED')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(normalized_translation_magnitudes, seds, c='red', label='SED vs Normalized Translation Magnitude')
    plt.xlabel('Normalized Translation Magnitude')
    plt.ylabel('SED')
    plt.legend()

    plt.suptitle('SED Error Analysis with Normalized Metrics')
    plt.show()

def move_based_on_image_0():
    for seq in os.listdir("sequences"):
        os.makedirs(os.path.join("sequences", seq, "image_1_moving"), exist_ok=True)

        src_dir = os.path.join("sequences", seq, "image_1")
        dst_dir = os.path.join("sequences", seq, "image_1_moving")

        if not os.path.exists(os.path.join("sequences", seq, "image_0_moving")):
            continue

        for img in os.listdir(os.path.join("sequences", seq, "image_0_moving")):
            if not os.path.exists(os.path.join("sequences", seq, "image_1", img)):
                continue
            os.rename(os.path.join(src_dir, img), os.path.join(dst_dir, img))

def bad_frame_to_txt():
    for seq in os.listdir('sequences'):
        if not os.path.exists(os.path.join('sequences', seq, 'image_0_moving')): continue

        for bad_frame_num in os.listdir(os.path.join('sequences', seq, 'image_0_moving')):
            txt_path = os.path.join('sequences', seq, 'bad_frames.txt')
            with open(txt_path, 'a') as f:
                f.write(f'{bad_frame_num} ')

def move_based_on_txt():
    no_move = ["01","04"]
    for seq in os.listdir('sequences'):
        if seq in no_move: continue

        txt_path = os.path.join('sequences', seq, 'bad_frames.txt')
        with open(txt_path, 'r') as file:
            bad_frames = file.readline().strip().split()

        src_dir0 = os.path.join("sequences", seq, "image_0")
        dst_dir0 = os.path.join("sequences", seq, "image_0_moving")
        src_dir1 = os.path.join("sequences", seq, "image_1")
        dst_dir1 = os.path.join("sequences", seq, "image_1_moving")

        os.makedirs(os.path.join(dst_dir0), exist_ok=True)
        os.makedirs(os.path.join(dst_dir1), exist_ok=True)

        for file_name in bad_frames:
            os.rename(os.path.join(src_dir0, file_name), os.path.join(dst_dir0, file_name))
            os.rename(os.path.join(src_dir1, file_name), os.path.join(dst_dir1, file_name))
            # print(os.path.join(src_dir0, file_name),  os.path.join(dst_dir0, file_name))

def update_epochs(file_path, increment):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    updated_lines = []
    
    for line in lines:
        if line.startswith("Epoch "):
            parts = line.split(" ")
            epoch_number = int(parts[1].split('/')[0])
            new_epoch_number = epoch_number + increment
            updated_line = line.replace(f"Epoch {epoch_number}/", f"Epoch {new_epoch_number}/")
            updated_lines.append(updated_line)

        else:
            updated_lines.append(line)

    with open(file_path, 'w') as file:
        file.writelines(updated_lines)


def move():
    # Define the source directory containing your files
    source_directory = "plots\\Stereo\\Winners"

    # Define the destination directories for each ratio
    ratio_directories = {
        "ratio_0.1": os.path.join(source_directory, "ratio_0.1"),
        "ratio_0.2": os.path.join(source_directory, "ratio_0.2"),
        "ratio_0.3": os.path.join(source_directory, "ratio_0.3")
    }

    # Create destination directories if they don't exist
    # for ratio_dir in ratio_directories.values():
    #     os.makedirs(ratio_dir, exist_ok=True)

    # Iterate over all files in the source directory
    for filename in os.listdir(source_directory):
        # Check for the ratio value in the filename and move accordingly
        for ratio, ratio_dir in ratio_directories.items():
            if ratio in filename:
                source_path = os.path.join(source_directory, filename)
                destination_path = os.path.join(ratio_dir, filename)
                # shutil.move(source_path, destination_path)
                print(f'from:\n{source_path}\nto:\n{destination_path}\n\n')
                # print(f"Moved {filename} to {ratio_dir}\n")
                break


def check_model_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return False
    
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        print(f"Error: The file {file_path} is empty.")
        return False
    
    print(f"The file {file_path} exists and is {file_size} bytes.")
    return True

    
def plot_errors():
    # PRETAINED VIT #
    # pretext_mean_alg_0 = [0.213333333, 0.283333333, 0.293333333, 0.363333333, 0.356666667, 0.373333333, 0.483333333, 0.6]
    # pretext_std_alg_0 = [0.015275252, 0.015275252, 0.023094011, 0.046188022, 0.049328829, 0.049328829, 0.073711148, 0.05]
    # pretext_mean_SED_0 =[0.193333333, 0.313333333, 0.323333333, 0.446666667, 0.44, 0.47, 0.762, 0.946666667]
    # pretext_std_SED_0 =[0.005773503, 0.032145503, 0.049328829, 0.068068593, 0.088881944, 0.112694277, 0.12701181, 0.083864971]
    # pretext_mean_RE1_0 = [0.049333333, 0.093333333, 0.064333333, 0.143333333, 0.14, 0.156333333, 0.273333333, 0.42]
    # pretext_std_RE1_0 = [0.001154701, 0.015275252, 0.044455971, 0.02081666, 0.036055513, 0.055410589, 0.055075705, 0.111355287]

    # pretext_mean_alg_4 = [0.23,  0.283333333, 0.32, 0.32, 0.403333333, 0.36, 0.462666667, 0.606666667]
    # pretext_std_alg_4 = [0.017320508, 0.011547005, 0.017320508, 0.01, 0.04163332, 0.07, 0.06466323, 0.092915732]
    # pretext_mean_SED_4 = [0.216666667, 0.313333333, 0.376666667, 0.383333333, 0.54, 0.446666667, 0.693333333, 1.01 ]
    # pretext_std_SED_4 = [0.015275252, 0.028867513, 0.058594653, 0.023094011, 0.09539392, 0.155026879, 0.187171935, 0.247588368]
    # pretext_mean_RE1_4 = [0.053333333, 0.09, 0.123333333, 0.12, 0.183333333,	 0.146666667, 0.246666667, 0.463333333]
    # pretext_std_RE1_4 = [0.005773503, 0.01,  0.028867513, 0.017320508,	0.023094011, 0.070237692, 0.066583281, 0.184481255]
	
    # pretext_mean_alg_8 = [0.23, 0.316666667, 0.32, 0.34, 0.372666667, 0.37, 0.453333333, 0.636666667]
    # pretext_mean_SED_8 = [0.213333333, 0.37, 0.36, 0.413, 0.466666667, 0.46, 0.653333333, 1.023333333]
    # pretext_mean_RE1_8 =[ 0.053333333, 0.11, 0.116, 0.136, 0.153333333, 0.16, 0.23, 0.486666667]
    # pretext_std_alg_8 = [0,   0.02081666, 0.017, 0.02, 0.014189198, 0.104403065, 0.085049005, 0.073711148]
    # pretext_std_SED_8 = [0.005773503, 0.036055513, 0.051, 0.02082, 0.030550505, 0.191572441, 0.175023808, 0.155026879]
    # pretext_std_RE1_8 = [0.005773503, 0.01, 0.02, 0.011547005, 0.011547005, 0.09539392, 0.075498344, 0.1530795]

    # KITTI Frozen: 0
    mean_alg_0 = [0.22,         0.3033,    0.25867,  0.3233,     0.36,      0.3933,    0.4667,        0.6267,   0.718]
    mean_SED_0 = [0.199666667,  0.353333,  0.272333, 0.38,       0.45,      0.506667,  0.67,          1.06,      1.25]
    mean_RE1_0 = [0.055,         0.106667, 0.073333, 0.116667,   0.146667,  0.173333,  0.226667,      0.486667,  0.583333333]
    std_alg_0 = [0.011547005,    0.025166, 0.025007, 0.015275,   0.078102,  0.070946,  0.025166,      0.141892,  0.147010204]
    std_SED_0 = [0.011547,       0.058595, 0.037233, 0.01,       0.130767,  0.138684,  0.06,          0.288444,  0.229128785]
    std_RE1_0 = [0,             0.028868, 0.011547,  0.005774,   0.049329,  0.077675,  0.030551,      0.225906,  0.140475383]

    # KITTI Frozen: 4
    mean_alg_4 = [0.216667, 0.303333, 0.248667, 0.306667, 0.366667, 0.373333, 0.446667, 0.57, 0.563333333]
    mean_SED_4 = [0.195667, 0.326667, 0.253333, 0.36, 0.456667, 0.476667, 0.626667, 0.92, 0.896]
    mean_RE1_4 = [0.05, 0.096, 0.066667, 0.113333, 0.143333, 0.163333, 0.22, 0.4, 0.36]
    std_alg_4 = [0.011547, 0.047258, 0.024194, 0.015275, 0.037859, 0.066583, 0.061101, 0.079373, 0.032145503]
    std_SED_4 = [0.005859, 0.068069, 0.025166, 0.03, 0.080829, 0.124231, 0.135031, 0.240624, 0.176714459]
    std_RE1_4 = [8.5e-18, 0.0151, 0.011547, 0.011547, 0.023094, 0.066583, 0.06, 0.160935, 0.13114877]

    # KITTI Frozen: 8
    mean_alg_8 = [0.213333, 0.283333, 0.26, 0.326667, 0.376667, 0.346667, 0.503333, 0.633333, 0.673333333]
    mean_SED_8 = [0.199, 0.31, 0.293333, 0.386667, 0.456667, 0.44, 0.763333, 1.053333, 1.121666667]
    mean_RE1_8 = [0.047, 0.091667, 0.083333, 0.123333, 0.153333, 0.14, 0.28, 0.483333, 0.518333333]
    std_alg_8 =  [0.025166115, 0.02081666, 0.03, 0.023094011, 0.015275252, 0.06350853, 0.056862407, 0.096090235, 0.045092498]
    std_SED_8 =  [0.037322915, 0.036055513, 0.055075705, 0.025166115, 0.032145503, 0.121243557, 0.167431578, 0.243378991, 0.07285831]
    std_RE1_8 =  [0.011269428, 0.017559423, 0.02081666,  0.005773503, 0.02081666, 0.060827625, 0.075498344, 0.191398363, 0.044859038]


    # KITTI Frozen: top 5
    mean_alg_5 =  [0.226666667, 0.315333333,	0.335333333,	0.331,	0.404666667,	0.452666667,	0.47,	0.686]
    mean_SED_5 = [0.21,	0.356666667, 0.38,	0.402,	0.52,	0.64,	0.676,	1.15666666]
    mean_RE1_5 =  [0.054666667,	0.106666667, 0.115,	0.128333333,	0.176666667,	0.223333333,	0.257666667,	0.53]
    std_alg_5 =  [0.005773503,	0.02532456,	0.009237604,	0.025357445,	0.054197171,	0.071143048,	0.052915026,	0.084071398]
    std_SED_5 =  [0,	0.045092498, 0.038935845,	0.031176915,	0.085440037,	0.165227116,	0.105910339,	0.160104133]
    std_RE1_5 =  [0.004163332,	0.02081666,	0.018027756, 0.01106044,	0.035118846,	0.077674535,	0.058620247,	0.051961524]

    # KITTI Resnet #
    resnet_mean_alg_0 = [0.246666667, 0.37, 0.378666667, 0.353333333, 0.423333333, 0.466666667, 0.42, 0.508333333]
    resnet_std_alg_0 =  [0.135769412, 0.05, 0.043650124, 0.065064071, 0.072341781, 	0.066583281, 0.173205081, 0.007637626]
    resnet_mean_SED_0 = [0.476666667, 0.463333333, 0.48, 0.413333333, 0.55, 0.586666667,  0.576666667, 0.696666667]
    resnet_std_SED_0 = [0.255799401, 0.090737717, 0.081853528, 0.105039675, 0.132287566, 0.092915732, 0.351046056, 0.032145503]
    resnet_mean_RE1_0 =  [0.083333333, 0.196666667, 0.166666667,	 0.146666667, 0.206666667,	 0.243333333, 0.21,	 0.291666667]
    resnet_std_RE1_0 = [0.082512625, 0.077674535, 0.030550505, 0.055075705, 0.061101009, 0.046188022, 0.147986486, 0.032532035]
    # KITTI: Clip 16
    clip_16_mean_alg =  [0.206666667, 0.245, 0.26, 0.295, 0.318, 0.266333333, 0.400333333, 0.46]
    clip_16_mean_SED =  [0.176666667, 0.233333333, 0.283333333, 0.33, 0.366, 0.346666667, 0.531, 0.653333333]
    clip_16_mean_RE1 =  [0.046666667, 0.064666667, 0.082666667, 0.1, 0.11, 0.083, 0.173333333, 0.24666666]
    clip_16_std_alg =  [0.04163332, 0.035355339, 0.026457513, 0.06, 0.051, 0.110545617, 0.025735838, 0.105830052]
    clip_16_std_SED =  [0.049328829, 0.05033223, 0.061101009, 0.11, 0.1, 0.092915732, 0.059253692, 0.192180471]
    clip_16_std_RE1 =  [0.015275252, 0.023180452, 0.02532456, 0.035, 0.036, 0.065551506, 0.025166115, 0.119303534]

    # Flying
    flying_clip_alg =  [0.36, 0.39, 1]
    flying_clip_SED = [0.71, 0.8, 3.34]
    flying_clip_RE1 = [0.14, 0.16, 1.19]
    flying_clip_16_alg =  [0.38, 0.34, 0.738]
    flying_clip_16_SED =  [0.73, 0.71, 2.34]
    flying_clip_16_RE1 = [0.14, 0.138, 0.689]
    flying_resnet_alg =  [0.41, 0.44, 0.78]
    flying_resnet_SED =  [0.84, 1, 2.57]
    flying_resnet_RE1 =  [0.175, 0.21, 0.73]

    os.makedirs('results', exist_ok=True)
    x_indices = range(len(mean_SED_0))  # For Frozen 0 (has an extra point)
    x_indices_flying = np.array(list(range(len(flying_clip_alg))))  
    xticks_labels = ['2166', '1082', '540', '405', '269', '161', '88', '47']  # 5 points for Frozen 0
    xticks_labels_flying = ['1431', '721', '88']  
    colors = ['Lightseagreen', 'burlywood', 'red']  # Red, Yellow, Cyan
    markers = ['o', 's', '^']  # Markers for each model
    linestyles = ['-', '--', ':']  # Line styles for each model    
    # x = np.arange(len(xticks_labels))  # x-coordinates for the groups
    # width = 0.25  # Width of each bar


    " capsize is the width of the error bars, linewidth is the width of the line, markersize is the size of the dots, alpha is the transparency of the error bars."


    fig7, axes7 = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    width = 0.17  # Width of each bar
    fig7.subplots_adjust(hspace=13)
    axes7[0].bar(x_indices_flying - width, flying_clip_alg, width, label='ViT-B/32', color=colors[0])
    axes7[0].bar(x_indices_flying, flying_clip_16_alg, width, label='ViT-B/16', color=colors[1])
    axes7[0].bar(x_indices_flying + width, flying_resnet_alg, width, label='ResNet-152', color=colors[2])
    axes7[0].set_title('ALG Metric of fine-tuned models on F estimation task using FlyingThings3D', fontsize=17)
    axes7[1].bar(x_indices_flying - width, flying_clip_SED, width, label='ViT-B/32', color=colors[0])
    axes7[1].bar(x_indices_flying, flying_clip_16_SED, width, label='ViT-B/16', color=colors[1])
    axes7[1].bar(x_indices_flying + width, flying_resnet_SED, width, label='ResNet-152', color=colors[2])
    axes7[1].set_title('SED Metric of fine-tuned models on F estimation task using FlyingThings3D', fontsize=17)
    axes7[2].bar(x_indices_flying - width, flying_clip_RE1, width, label='ViT-B/32', color=colors[0])
    axes7[2].bar(x_indices_flying, flying_clip_16_RE1, width, label='ViT-B/16', color=colors[1])
    axes7[2].bar(x_indices_flying + width, flying_resnet_RE1, width, label='ResNet-152', color=colors[2])
    axes7[2].set_title('SAM Metric of fine-tuned models on F estimation task using FlyingThings3D', fontsize=17)

    for ax in axes7:
        ax.set_xlabel('Number of Training Samples', fontsize=14)      
        ax.set_ylabel('Mean Value ± STD', fontsize=14)  
        ax.legend(loc='upper left', fontsize=14)
        ax.set_xticks(range(len(xticks_labels_flying)), labels=xticks_labels_flying) 
        ax.tick_params(axis='both', labelbottom=True, labelsize=14)  # Force x-tick labels to be shown
        ax.grid(True, linestyle='-', color='#d3d3d3')
    plt.tight_layout()
    fig7.savefig('results/models_flying.png')



    fig5, axes5 = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    fig5.subplots_adjust(hspace=13)  # Add space between subplots
    axes5[0].errorbar(x_indices, mean_alg_0, yerr=std_alg_0, marker=markers[0], color=colors[0], linestyle=linestyles[0], label='ViT-B/32', capsize=8, linewidth=2, markersize=7)
    axes5[0].errorbar(x_indices, clip_16_mean_alg, yerr=clip_16_std_alg, marker=markers[1], color=colors[1], linestyle=linestyles[1], label='ViT-B/16', capsize=8, linewidth=2, markersize=7)
    axes5[0].errorbar(x_indices, resnet_mean_alg_0, yerr=resnet_std_alg_0, marker=markers[2], color=colors[2], linestyle=linestyles[2], label='ResNet-152', capsize=8, linewidth=2, markersize=7)
    axes5[0].set_title('ALG Metric of fine-tuned models on F estimation task using KITTI', fontsize=17)
    axes5[1].errorbar(x_indices, mean_SED_0, yerr=std_SED_0, marker=markers[0], color=colors[0], linestyle=linestyles[0], label='ViT-B/32', capsize=8, linewidth=2, markersize=7)
    axes5[1].errorbar(x_indices, clip_16_mean_SED, yerr=clip_16_std_SED, marker=markers[1], color=colors[1], linestyle=linestyles[1], label='ViT-B/16', capsize=8, linewidth=2, markersize=7)
    axes5[1].errorbar(x_indices, resnet_mean_SED_0, yerr=resnet_std_SED_0, marker=markers[2], color=colors[2], linestyle=linestyles[2], label='ResNet-152', capsize=8, linewidth=2, markersize=7)
    axes5[1].set_title('SED Metric of fine-tuned models on F estimation task using KITTI', fontsize=17)
    axes5[2].errorbar(x_indices, mean_RE1_0, yerr=std_RE1_0, marker=markers[0], color=colors[0], linestyle=linestyles[0], label='ViT-B/32', capsize=8, linewidth=2, markersize=7)
    axes5[2].errorbar(x_indices, clip_16_mean_RE1, yerr=clip_16_std_RE1, marker=markers[1], color=colors[1], linestyle=linestyles[1], label='ViT-B/16', capsize=8, linewidth=2, markersize=7)
    axes5[2].errorbar(x_indices, resnet_mean_RE1_0, yerr=resnet_std_RE1_0, marker=markers[2], color=colors[2], linestyle=linestyles[2], label='ResNet-152', capsize=8, linewidth=2, markersize=7)
    axes5[2].set_title('SAM Metric of fine-tuned models on F estimation task using KITTI', fontsize=17)
    for ax in axes5:
        ax.set_xlabel('Number of Training Samples', fontsize=14)        
        ax.set_ylabel('Mean Value ± STD', fontsize=14)
        ax.legend(loc='upper left', fontsize=14)
        ax.grid(True, linestyle='-', color='#d3d3d3')
        ax.set_xticks(range(len(xticks_labels)), labels=xticks_labels) 
        ax.tick_params(axis='both', labelbottom=True, labelsize=14)  # Force x-tick labels to be shown
    plt.tight_layout()  # Adjust the layout to make room for the title
    fig5.savefig('results/models_kitti.png')


    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    fig1.subplots_adjust(hspace=13)  # Add space between subplots   
    axes1[0].errorbar(x_indices, mean_alg_0, yerr=std_alg_0, marker=markers[0], color=colors[0], linestyle=linestyles[0], label='ALG 0 bottom frozen layers', capsize=8, linewidth=2, markersize=7)
    axes1[0].errorbar(x_indices, mean_alg_4, yerr=std_alg_4, marker=markers[1], color=colors[1], linestyle=linestyles[1], label='ALG 4 bottom frozen layers', capsize=8, linewidth=2, markersize=7)
    axes1[0].errorbar(x_indices, mean_alg_8, yerr=std_alg_8, marker=markers[2], color=colors[2], linestyle=linestyles[2], label='ALG 8 bottom frozen layers', capsize=8, linewidth=2, markersize=7)
    axes1[0].set_title('ALG Metric of ViT-B/32 freezing bottom layers on F estimation task using KITTI', fontsize=17)
    axes1[1].errorbar(x_indices, mean_SED_0, yerr=std_SED_0, marker=markers[0], color=colors[0], linestyle=linestyles[0], label='SED 0 bottom frozen layers', capsize=8, linewidth=2, markersize=7)
    axes1[1].errorbar(x_indices, mean_SED_4, yerr=std_SED_4, marker=markers[1], color=colors[1], linestyle=linestyles[1], label='SED 4 bottom frozen layers', capsize=8, linewidth=2, markersize=7)
    axes1[1].errorbar(x_indices, mean_SED_8, yerr=std_SED_8, marker=markers[2], color=colors[2], linestyle=linestyles[2], label='SED 8 bottom frozen layers', capsize=8, linewidth=2, markersize=7)
    axes1[1].set_title('SED Metric of ViT-B/32 freezing bottom layers on F estimation task using KITTI', fontsize=17)
    axes1[2].errorbar(x_indices, mean_RE1_0, yerr=std_RE1_0, marker=markers[0], color=colors[0], linestyle=linestyles[0], label='SAM 0 bottom frozen layers', capsize=8, linewidth=2, markersize=7)
    axes1[2].errorbar(x_indices, mean_RE1_4, yerr=std_RE1_4, marker=markers[1], color=colors[1], linestyle=linestyles[1], label='SAM 4 bottom frozen layers', capsize=8, linewidth=2, markersize=7)
    axes1[2].errorbar(x_indices, mean_RE1_8, yerr=std_RE1_8, marker=markers[2], color=colors[2], linestyle=linestyles[2], label='SAM 8 bottom frozen layers', capsize=8, linewidth=2, markersize=7)
    axes1[2].set_title('SAM Metric of ViT-B/32 freezing bottom layers on F estimation task using KITTI', fontsize=17)
    for ax in axes1:
        ax.set_xlabel('Number of Training Samples', fontsize=14) 
        ax.set_ylabel('Mean Value ± STD', fontsize=14)       
        ax.legend(loc='upper left', fontsize=14)
        ax.grid(True, linestyle='-', color='#d3d3d3')
        ax.set_xticks(range(len(xticks_labels)), labels=xticks_labels) 
        ax.tick_params(axis='both', labelbottom=True, labelsize=14)  # Force x-tick labels to be shown
    plt.tight_layout()  # Adjust the layout to make room for the title
    fig1.savefig('results/Frozen_low.png')


    fig4, axes4 = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    fig4.subplots_adjust(hspace=13)  # Add space between subplots   
    axes4[0].errorbar(x_indices, mean_alg_0, yerr=std_alg_0, marker=markers[0], color=colors[0], linestyle=linestyles[0], label='ALG 0 top frozen layers', capsize=8, linewidth=2, markersize=7)
    axes4[0].errorbar(x_indices, mean_alg_5, yerr=std_alg_5, marker=markers[2], color=colors[2], linestyle=linestyles[2], label='ALG 5 top frozen layers', capsize=8, linewidth=2, markersize=7)
    axes4[0].set_title('ALG Metric of ViT-B/32 freezing top layers on F estimation task using KITTI', fontsize=17)
    axes4[1].errorbar(x_indices, mean_SED_0, yerr=std_SED_0, marker=markers[0], color=colors[0], linestyle=linestyles[0], label='SED 0 top frozen layers', capsize=8, linewidth=2, markersize=7)
    axes4[1].errorbar(x_indices, mean_SED_5, yerr=std_SED_5, marker=markers[2], color=colors[2], linestyle=linestyles[2], label='SED 5 top frozen layers', capsize=8, linewidth=2, markersize=7)
    axes4[1].set_title('SED Metric of ViT-B/32 freezing top layers on F estimation task using KITTI', fontsize=17)
    axes4[2].errorbar(x_indices, mean_RE1_0, yerr=std_RE1_0, marker=markers[0], color=colors[0], linestyle=linestyles[0], label='SAM 0 top frozen layers', capsize=8, linewidth=2, markersize=7)
    axes4[2].errorbar(x_indices, mean_RE1_5, yerr=std_RE1_5, marker=markers[2], color=colors[2], linestyle=linestyles[2], label='SAM 5 top frozen layers', capsize=8, linewidth=2, markersize=7)
    axes4[2].set_title('SAM Metric of ViT-B/32 freezing top layers on F estimation task using KITTI', fontsize=17)
    for ax in axes4:
        ax.set_xlabel('Number of Training Samples', fontsize=14)  
        ax.set_ylabel('Mean Value ± STD', fontsize=14)      
        ax.legend(loc='upper left', fontsize=14)
        ax.grid(True, linestyle='-', color='#d3d3d3')
        ax.set_xticks(range(len(xticks_labels)), labels=xticks_labels) 
        ax.tick_params(axis='both', labelbottom=True, labelsize=14)  # Force x-tick labels to be shown
    plt.tight_layout()  # Adjust the layout to make room for the title
    fig4.savefig('results/Frozen_high.png')
   

    # fig3=plt.figure(3, figsize=(11, 6))
    # plt.errorbar(x_indices, mean_alg_0, yerr=std_alg_0, marker='o', color='blue', linestyle=':', label='ALG ViT-B/32', capsize=4, linewidth=1, markersize=2) 
    # plt.errorbar(x_indices, mean_SED_0, yerr=std_SED_0, marker='o', color='blue', linestyle='-', label='SED ViT-B/32', capsize=4, linewidth=1, markersize=2)
    # plt.errorbar(x_indices, mean_RE1_0, yerr=std_RE1_0, marker='o', color='blue', linestyle='--', label='SAM ViT-B/32', capsize=4, linewidth=1, markersize=2)
    # plt.errorbar(x_indices, pretext_mean_alg_0, yerr=pretext_std_alg_0, marker='o', color='orange', linestyle=':', label='ALG Pretext ViT-B/32', capsize=4, linewidth=1, markersize=2)
    # plt.errorbar(x_indices, pretext_mean_SED_0, yerr=pretext_std_SED_0, marker='o', color='orange', linestyle='-', label='SED Pretext ViT-B/32', capsize=4, linewidth=1, markersize=2)
    # plt.errorbar(x_indices, pretext_mean_RE1_0, yerr=pretext_std_RE1_0, marker='o', color='orange', linestyle='--', label='SAM Pretext ViT-B/32', capsize=4, linewidth=1, markersize=2)
    # plt.title('Fine tuned pretext task ViT and original ViT on F-Matrix task using KITTI dataset')
    # plt.xlabel('Number of training samples')
    # plt.ylabel('Mean Value ± STD')
    # plt.xticks(range(len(xticks_labels)), labels=xticks_labels)  
    # plt.legend()
    # plt.grid(True)
    # fig3.savefig('results/pretext.png')



def sed_distance_gt_FM():
    train_loader = get_dataloader_FM(batch_size=1)

    epoch_stats = {"test_algebraic_pred": torch.tensor(0), "test_algebraic_sqr_pred": torch.tensor(0), "test_RE1_pred": torch.tensor(0), "test_SED_pred": torch.tensor(0),
                   "test_algebraic_truth": torch.tensor(0), "test_algebraic_sqr_truth": torch.tensor(0), "test_RE1_truth": torch.tensor(0), "test_SED_truth": torch.tensor(0),
                   "test_loss": torch.tensor(0), "test_labels": torch.tensor([]), "test_outputs": torch.tensor([])}
    
    for i, (img1, img2, label, pts1, pts2, _) in enumerate(train_loader):
        img1, img2, label, pts1, pts2 = img1.to(device), img2.to(device), label.to(device), pts1.to(device), pts2.to(device)

        update_epoch_stats(epoch_stats, img1.detach(), img2.detach(), label.detach(), label.detach(), pts1, pts2, "", data_type="test")
    
        pts1 = pts1[0].cpu().numpy()
        pts2 = pts2[0].cpu().numpy()
        # Convert grayscale tensors to numpy arrays for matplotlib
        img0_np = reverse_transforms(img1[0].cpu(), mean=norm_mean.cpu(), std=norm_std.cpu())  # shape (H, W, C)
        img1_np = reverse_transforms(img2[0].cpu(), mean=norm_mean.cpu(), std=norm_std.cpu())  # shape (H, W, C)
        print(img1_np.shape)    
        img0_pts = img0_np.copy()
        img1_pts = img1_np.copy()
        for point in pts1:
            if point[0] == 0 and point[1] == 0: continue
            img0_pts = cv2.circle(img0_pts, (int(point[0]), int(point[1])), 3, (20, 20, 160), -1)
            
        for point in pts2:
            if point[0] == 0 and point[1] == 0: continue
            img1_pts = cv2.circle(img1_pts, (int(point[0]), int(point[1])), 3, (20, 20, 160), -1)

        # Concatenate images horizontally
        combined_image = np.hstack((img0_pts, img1_pts))

        os.makedirs(f'gt_epilines/FM', exist_ok=True)
        cv2.imwrite(f'gt_epilines/FM/gt_{i}.png', combined_image)
        
        break

    print(f'\ntest_algebraic_pred: {epoch_stats["test_algebraic_pred"]/(i+1)}')
    print(f'test_algebraic_sqr_pred: {epoch_stats["test_algebraic_sqr_pred"]/(i+1)}')
    print(f'test_RE1_pred: {epoch_stats["test_RE1_pred"]/(i+1)}')
    print(f'test_SED_pred: {epoch_stats["test_SED_pred"]/(i+1)}')
    print()

import struct

def read_cam_file(file_path):
    with open(file_path, "rb") as f:
        # Step 1: Read and check the "PIEH" tag (4 bytes, float format)
        tag_data = f.read(4)
        tag = struct.unpack("f", tag_data)[0]
        
        # Check if the tag matches 202021.25
        if tag != 202021.25:
            raise ValueError("Invalid .cam file: Tag does not match expected value")

        # Step 2: Read the intrinsic 3x3 matrix (9 floats, 64-bit)
        intrinsic_data = f.read(9 * 8)
        intrinsic_matrix = struct.unpack("9d", intrinsic_data)  # 'd' for double (64-bit float)
        intrinsic_matrix = np.array(intrinsic_matrix).reshape((3, 3))

        # Step 3: Read the extrinsic 3x4 matrix (12 floats, 64-bit)
        extrinsic_data = f.read(12 * 8)
        extrinsic_matrix = struct.unpack("12d", extrinsic_data)
        extrinsic_matrix = np.array(extrinsic_matrix).reshape((3, 4))

    return intrinsic_matrix, extrinsic_matrix


def delete_odd_files(folder_path):
    right_path = os.path.join(folder_path, "right")
    left_path = os.path.join(folder_path, "left")
    for filename in os.listdir(right_path):
        file_number = int(filename.split('.')[0])  # Convert to an integer
        if file_number % 2 != 0:  # Check if the number is odd
            right_file_path = os.path.join(right_path, filename)
            left_file_path = os.path.join(left_path, filename)
            try:
                # os.remove(right_file_path)  
                print(f"Deleted: {right_file_path}")
            except OSError as e:
                print(f"Error deleting {right_file_path}: {e}")
            try:
                # os.remove(left_file_path)
                print(f"Deleted: {left_file_path}")
            except OSError as e:    
                print(f"Error deleting {left_file_path}: {e}")

def test_trained(pretrained_model):
    " Only need to change the data type in params i.e SCENEFLOW, KITTI.. "
    batch_size=1

    train_loader, val_loader, test_loader = get_data_loaders(train_size=0.004, part='head', batch_size=batch_size)

    model = FMatrixRegressor(lr=LR[0], batch_size=batch_size, L2_coeff=L2_COEFF, huber_coeff=HUBER_COEFF, trained_vit=TRAINED_VIT, frozen_layers=0, pretrained_path=pretrained_model).to(device)

    for img1, img2, label, pts1, pts2, _,  in test_loader:
        img1, img2, label, pts1, pts2 = img1.to(device), img2.to(device), label.to(device), pts1.to(device), pts2.to(device)
        F_est = model.forward(img1, img2)
        print(f'label: {label}')
        print(f'F_est: {F_est}')
        break
    # model.test(test_loader=test_loader, write=False)
    # print(model.start_epoch)
    # print(pretrained_model)

def test_specific_F(F):
    batch_size=1
    _, _, test_loader = get_data_loaders(train_size=0.004, part='head', batch_size=batch_size)

    avg_sed = 0
    for i, (img1, img2, label, pts1, pts2, _) in enumerate(test_loader):
        ep = EpipolarGeometry(None, None, F, pts1, pts2)
        sed = ep.get_mean_SED_distance()

        avg_sed += sed

    avg_sed = avg_sed / (i+1)
    print(f'\nAverage SED: {avg_sed}, {i+1}')
    return avg_sed

def RANSAC():
    batch_size=1
    _, _, test_loader = get_data_loaders(train_size=0.004, part='head', batch_size=batch_size)

    avg_sed = 0
    did = 1
    for i, (_, _, _, pts1, pts2, _) in enumerate(test_loader):
        if pts1.shape[1] < 10: continue
        pts1_np = pts1.squeeze(0).cpu().numpy()[:,:2]
        pts2_np = pts2.squeeze(0).cpu().numpy()[:,:2]

        F, mask = cv2.findFundamentalMat(pts1_np, pts2_np, cv2.FM_RANSAC, 1, 0.99)
        F = torch.from_numpy(F).float().unsqueeze(0).to(device)
        print(F)
        ep = EpipolarGeometry(None, None, F, pts1, pts2)
        sed = ep.get_mean_SED_distance()

        avg_sed += sed
        did += 1
        # if i > 200: break

    avg_sed = avg_sed / did
    print(f'\nAverage SED: {avg_sed}, {did}')
    return avg_sed

def avg_trained():
    batch_size=1
    _, _, test_loader = get_data_loaders(train_size=0.004, part='head', batch_size=batch_size)

    pretrained_path = "plots/Stereo/Winners/SED_0.5__L2_1__huber_1__lr_0.0001__conv__CLIP__use_reconstruction_True/BS_8__ratio_0.2__mid__frozen_0"
    model = FMatrixRegressor(lr=LR[0], batch_size=batch_size, L2_coeff=L2_COEFF, huber_coeff=HUBER_COEFF, trained_vit=TRAINED_VIT, frozen_layers=0, pretrained_path=pretrained_path).to(device)
    
    avg_F = torch.zeros((3, 3)).to(device)
    for img1, img2, _, _, _, _ in test_loader:
        img1, img2 = img1.to(device), img2.to(device)

        output = model.forward(img1, img2).detach()
        print(output)
        return
        avg_F += output.squeeze(0)
    
    avg_F = avg_F / len(test_loader)
    print(avg_F)

    # Got:
    # avg_F = torch.tensor([[[-5.6917e-06,  2.5964e-03, -2.0555e-01],
    #                   [-2.5585e-03,  1.0635e-04, -6.8064e-01],
    #                   [ 2.0113e-01,  6.7193e-01,  4.3438e-02]]]).to(device)

import matplotlib
matplotlib.use('Agg') # If want to show images then disable this
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    p = "plots/Stereo/Winners/SED_0.5__L2_1__huber_1__lr_0.0001__conv__CLIP_16__use_reconstruction_True/BS_8__ratio_0.2__head__frozen_0"

    test_trained(p)
    # plot_errors()
    # RANSAC()
    # avg_trained()
    # test_specific_F(avg_F)
