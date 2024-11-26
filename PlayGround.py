import time

import torchvision
from Dataset_FM import get_dataloader_FM
from utils import print_and_write, reverse_transforms
from FunMatrix import EpipolarGeometry, compute_fundamental, get_F, update_epoch_stats
from FMatrixRegressor import FMatrixRegressor
from Dataset import get_data_loaders
from params import LR, device, norm_mean, norm_std

import cv2
import numpy as np
import matplotlib.pyplot as plt
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
    train_loader, val_loader, test_loader = get_data_loaders(train_size=10, batch_size=1)
    total_sed = 0

    for i, (img1, img2, label, pts1, pts2, seq_name) in enumerate(test_loader):
        pts1 = pts1[0].cpu().numpy()
        pts2 = pts2[0].cpu().numpy()
        # Convert grayscale tensors to numpy arrays for matplotlib
        img0_np = reverse_transforms(img1[0].cpu(), mean=norm_mean.cpu(), std=norm_std.cpu())  # shape (H, W, C)
        img1_np = reverse_transforms(img2[0].cpu(), mean=norm_mean.cpu(), std=norm_std.cpu())  # shape (H, W, C)
        
        img0_pts = img0_np.copy()
        img1_pts = img1_np.copy()
        for i,point in enumerate(pts1):
            if i == 30: break
            if point[0] == 0 and point[1] == 0: continue
            img0_pts = cv2.circle(img0_pts, (int(point[0]), int(point[1])), 2, (20, 20, 160), -1)
            
        for i, point in enumerate(pts2):
            if i == 30: break
            if point[0] == 0 and point[1] == 0: continue
            img1_pts = cv2.circle(img1_pts, (int(point[0]), int(point[1])), 2, (20, 20, 160), -1)

        # Create padding (e.g., 10-pixel wide, white vertical strip)
        padding = 255 * np.zeros((img0_pts.shape[0], 30, 3), dtype=np.uint8)  # 10-pixel wide white space

        # Combine the two images with padding in between
        combined_image = np.hstack((img0_pts, padding, img1_pts))

        os.makedirs(f'gt_epilines/RealEstate_after_transform/{seq_name[0]}', exist_ok=True)
        cv2.imwrite(f'gt_epilines/RealEstate_after_transform/{seq_name[0]}/gt_{i}.png', combined_image)

        if i == 100: break
        
    total_sed /= i
    print(f'SED distance: {total_sed}') 

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


def vis_trained(plots_path):
    model = FMatrixRegressor(lr_vit=2e-5, lr_mlp=2e-5, pretrained_path=plots_path)
    
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=1)
    for i, (img1, img2, label, pts1, pts2, seq_name) in enumerate(val_loader):
        img1, img2 = img1.to(device), img2.to(device)
        output = model.forward(img1, img2)

        epipolar_geo = EpipolarGeometry(img1[0], img2[0], output[0].detach(), pts1=pts1[0], pts2=pts2[0])
        epipolar_geo.visualize(idx=i, epipolar_lines_path=os.path.join("predicted_RealEstate", seq_name[0]))

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

        pts1 = pts1[0].cpu().numpy()
        pts2 = pts2[0].cpu().numpy()
        
        # Convert grayscale tensors to numpy arrays for matplotlib
        img0_np = reverse_transforms(img1[0].cpu(), mean=norm_mean.cpu(), std=norm_std.cpu())  # shape (H, W, C)
        img1_np = reverse_transforms(img2[0].cpu(), mean=norm_mean.cpu(), std=norm_std.cpu())  # shape (H, W, C)

        img0_np = cv2.cvtColor(img0_np, cv2.COLOR_GRAY2RGB)
        img1_np = cv2.cvtColor(img1_np, cv2.COLOR_GRAY2RGB)

        img0_pts = img0_np.copy()
        img1_pts = img1_np.copy()
        for i,point in enumerate(pts1):
            if i == 30: break
            if point[0] == 0 and point[1] == 0: continue
            img0_pts = cv2.circle(img0_pts, (int(point[0]), int(point[1])), 2, (20, 20, 160), -1)
            
        for i, point in enumerate(pts2):
            if i == 30: break
            if point[0] == 0 and point[1] == 0: continue
            img1_pts = cv2.circle(img1_pts, (int(point[0]), int(point[1])), 2, (20, 20, 160), -1)

        # Create padding (e.g., 10-pixel wide, white vertical strip)
        padding = 255 * np.zeros((img0_pts.shape[0], 30, 3), dtype=np.uint8)  # 10-pixel wide white space

        # Combine the two images with padding in between
        combined_image = np.hstack((img0_pts, padding, img1_pts))

        os.makedirs(f'gt_epilines/monkaa/{seq_name[0]}', exist_ok=True)
        cv2.imwrite(f'gt_epilines/monkaa/{seq_name[0]}/gt_{i}.png', combined_image)

        update_epoch_stats(epoch_stats, img1.detach(), img2.detach(), label.detach(), output, pts1, pts2, data_type="test")
        if i == 10: break
    

    print(f"""SED distance: {epoch_stats["test_SED_pred"]/(i+1)}
Algebraic distance: {epoch_stats["test_algebraic_pred"]/(i+1)}
RE1 distance: {epoch_stats["test_RE1_pred"]/(i+1)}

SED distance truth: {epoch_stats["test_SED_truth"]/(i+1)}
Algebraic distance truth: {epoch_stats["test_algebraic_truth"]/(i+1)}
RE1 distance truth: {epoch_stats["test_RE1_truth"]/(i+1)}""")


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


def extend_runs(batch_size, lr, lr_decay, weight_decay, L2_coeff, huber_coeff, data_ratio):
    train_loader, val_loader, test_loader = get_data_loaders(data_ratio, batch_size)
    with open('runs.txt', 'r') as f:
        for pretrained_path in f:
            pretrained_path = pretrained_path.strip()
            if not os.path.exists(pretrained_path):
                    print("problema with path: " + pretrained_path)
                    continue
            model = FMatrixRegressor(lr=lr, lr_decay=lr_decay, wd=weight_decay, batch_size=batch_size, L2_coeff=L2_coeff, huber_coeff=huber_coeff, pretrained_path=pretrained_path).to(device)
            print_and_write(f"##### CONTINUE TRAINING #####\n\n", model.plots_path)
            model.train_model(train_loader, val_loader, test_loader)

            torch.cuda.empty_cache()
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
    # # Frozen: 0
    mean_alg_0 = [0.275666667, 0.386333333, 0.333333333, 0.463333333, 0.576666667, 0.538333333]
    std_alg_0 = [0.0246, 0.0718, 0.0252, 0.0252, 0.0611, 0.1173]
    mean_SED_0 = [0.28, 0.493333333, 0.4, 0.613333333, 0.84, 0.77]
    std_SED_0 = [0.0265, 0.1447, 0.06, 0.0208, 0.1353, 0.283]
    mean_RE1_0 = [0.079666667, 0.183333333, 0.13, 0.236666667, 0.333333333, 0.356666667]
    std_RE1_0 = [0.0095, 0.0924, 0.02, 0.0058, 0.0603, 0.2043]

    mean_alg_4 = [0.26833333, 0.43, 0.34433333, 0.49666667, 0.56333333, 0.54333333]
    std_alg_4 = [0.02466441, 0.03605551, 0.01250333, 0.09451631, 0.04932883, 0.10692677]
    mean_SED_4 = [0.26766667, 0.59, 0.41666667, 0.70333333, 0.84, 0.78333833]
    std_SED_4 = [0.02400694, 0.07549834, 0.04163332, 0.17156146, 0.09848858, 0.19553846]
    mean_RE1_4 = [0.07466667, 0.21333333, 0.13333333, 0.27333333, 0.36, 0.343333]
    std_RE1_4 = [0.00493288, 0.04041452, 0.02516611, 0.07234178, 0.05291503, 0.13796135]

    mean_alg_8 = [0.251333333, 0.43, 0.373333333, 0.52, 0.556666667, 0.553333333]
    std_alg_8 = [0.02967041, 0.01, 0.04163332, 0.06244998, 0.04725816, 0.11150486]
    mean_SED_8 = [0.251666667, 0.576, 0.481333333, 0.733333333, 0.823333333, 0.8]
    std_SED_8 = [0.04964205, 0.002915, 0.07710599, 0.09073772, 0.106044, 0.24576411]
    mean_RE1_8 = [0.071333333, 0.213333333, 0.17, 0.293333333, 0.366666667, 0.35]
    std_RE1_8 = [0.01761628, 0.02516611, 0.04358899, 0.05033223, 0.04163332, 0.17578396]

    # PRETAINED VIT #
    pretext_mean_alg_0 = [0.263333333, 0.390333, 0.473, 0.513333333, 0.528333333, 0.556333333]
    pretext_std_alg_0 = [0.0321, 0.087, 0.0878, 0.0586, 0.0679, 0.1245]
    pretext_mean_SED_0 = [0.258, 0.506, 0.665333333, 0.724666667, 0.752333333, 0.823333333]
    pretext_std_SED_0 = [0.0365, 0.0898, 0.205, 0.0609, 0.1234, 0.2603]
    pretext_mean_RE1_0 = [0.321, 0.179, 0.264333333, 0.313666667, 0.315666667, 0.375666667]
    pretext_std_RE1_0 = [0.4495, 0.0347, 0.0955, 0.049, 0.0915, 0.2047]

    pretext_mean_alg_4 = [0.27433333, 0.41, 0.51333333, 0.52333333, 0.60733333, 0.54666667]
    pretext_std_alg_4 = [0.01628906, 0.04, 0.04454586, 0.05859865, 0.03523256, 0.16010813]
    pretext_mean_SED_4 = [0.28366667, 0.54333333, 0.74533333, 0.75333333, 0.885, 0.82666667]
    pretext_std_SED_4 = [0.02345918, 0.11372481, 0.10929471, 0.14294521, 0.10331989, 0.35076108]
    pretext_mean_RE1_4 = [0.08433333, 0.21, 0.297, 0.31866667, 0.39366667, 0.389]
    pretext_std_RE1_4 = [0.00971253, 0.08717798, 0.03819686, 0.06344551, 0.04324735, 0.2606396]

    pretext_mean_alg_8 = [0.265666667, 0.423333333, 0.486666667, 0.566666667, 0.593333333, 0.66]
    pretext_std_alg_8 = [0.02223361, 0.03511885, 0.03511885, 0.05507571, 0.005505, 0.08185353]
    pretext_mean_SED_8 = [0.275, 0.557333333, 0.681666667, 0.845333333, 0.893333333, 1.07]
    pretext_std_SED_8 = [0.031, 0.08967348, 0.10774198, 0.11360164, 0.05131601, 0.23]
    pretext_mean_RE1_8 = [0.079666667, 0.2212, 0.266666667, 0.365333333, 0.498333333, 0.523]
    pretext_std_RE1_8 = [0.01250333, 0.06982922, 0.05773503, 0.0080829, 0.04843897, 0.21488804]

    # RESNET #
    resnet_mean_alg_0 = [0.267, 0.410333333, 0.491333333, 0.54, 0.576666667, 0.570333333]
    resnet_std_alg_0 = [0.131, 0.0542, 0.0776, 0.0529, 0.1159, 0.1124]
    resnet_mean_SED_0 = [0.523333333, 0.518666667, 0.6432, 0.72, 0.783333333, 0.763666667]
    resnet_std_SED_0 = [0.2351, 0.0924, 0.1414, 0.0721, 0.185, 0.2048]
    resnet_mean_RE1_0 = [0.114033333, 0.222333333, 0.27, 0.33, 0.350666667, 0.371333333]
    resnet_std_RE1_0 = [0.1106, 0.0686, 0.0755, 0.0557, 0.0853, 0.1634]


    x_indices = range(len(mean_SED_0))  # For Frozen 0 (has an extra point)
    xticks_labels = ['2166', '1082', '540', '405', '269', '161']  # 5 points for Frozen 0
    x = np.arange(len(xticks_labels))  # x-coordinates for the groups
    width = 0.25  # Width of each bar

    # plt.errorbar(x_indices, mean_SED_0, yerr=std_SED_0, marker='o', color='blue', linestyle='-', label='SED Frozen 0', capsize=5)
    # plt.errorbar(x_indices, mean_SED_4, yerr=std_SED_4, marker='o', color='green', linestyle='-', label='SED Frozen 4', capsize=5)
    # plt.errorbar(x_indices, mean_SED_8, yerr=std_SED_8, marker='o', color='orange', linestyle='-', label='SED Frozen 8', capsize=5)
    # plt.title('SED comparison of original model with different frozen layers')
    # plt.bar(x, mean_SED_0, width, yerr=std_SED_0, capsize=5, label='SED Frozen 0', alpha=0.8, color='blue')
    # plt.bar(x - width, mean_SED_4, width, yerr=std_SED_4, capsize=5, label='SED Frozen 4', alpha=0.8, color='green')
    # plt.bar(x + width, mean_SED_8, width, yerr=std_SED_8, capsize=5, label='SED Frozen 8', alpha=0.8, color='orange')
    # plt.title('Barplot SED comparison of original model with different frozen layers')

    # plt.errorbar(x_indices, pretext_mean_SED_0, yerr=pretext_std_SED_0, marker='o', color='blue', linestyle='-', label='SED Frozen 0', capsize=5)
    # plt.errorbar(x_indices, pretext_mean_SED_4, yerr=pretext_std_SED_4, marker='o', color='green', linestyle='-', label='SED Frozen 4', capsize=5)
    # plt.errorbar(x_indices, pretext_mean_SED_8, yerr=pretext_std_SED_8, marker='o', color='orange', linestyle='-', label='SED Frozen 8', capsize=5)
    # plt.title('SED comparison of pretext model with different frozen layers')
    plt.bar(x, pretext_mean_SED_0, width, yerr=pretext_std_SED_0, capsize=5, label='SED Frozen 0', alpha=0.8, color='blue')
    plt.bar(x - width, pretext_mean_SED_4, width, yerr=pretext_std_SED_4, capsize=5, label='SED Frozen 4', alpha=0.8, color='green')
    plt.bar(x + width, pretext_mean_SED_8, width, yerr=pretext_std_SED_8, capsize=5, label='SED Frozen 8', alpha=0.8, color='orange')
    plt.title('Barplot SED comparison of pretext model with different frozen layers')

    # plt.errorbar(x_indices, mean_SED_0, yerr=std_SED_0, marker='o', color='blue', linestyle='-', label='SED Original Frozen 0', capsize=5)
    # plt.errorbar(x_indices, pretext_mean_SED_0, yerr=pretext_std_SED_0, marker='o', color='green', linestyle='-', label='SED Pretext Frozen 0', capsize=5)
    # plt.errorbar(x_indices, resnet_mean_SED_0, yerr=resnet_std_SED_0, marker='o', color='orange', linestyle='-', label='SED ResNet', capsize=5)
    # plt.title('SED comparison of original, pretext and resnet models')
    # plt.bar(x, mean_SED_0, width, yerr=std_SED_0, capsize=5, label='Original model', alpha=0.8, color='blue')
    # plt.bar(x - width, pretext_mean_SED_0, width, yerr=pretext_std_SED_0, capsize=5, label='Pretext model', alpha=0.8, color='green')
    # plt.bar(x + width, resnet_mean_SED_0, width, yerr=resnet_std_SED_0, capsize=5, label='ResNet', alpha=0.8, color='orange')
    # plt.title('Barplot SED comparison of original, pretext and resnet models')
              
    plt.xlabel('Data Points')
    plt.ylabel('Mean Value ± STD')
    plt.xticks(range(len(xticks_labels)), labels=xticks_labels)  # Adjusting X-axis labels for Frozen 0
    plt.legend()
    plt.grid(True)

    plt.savefig("5")

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



if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    sed_distance_trained()