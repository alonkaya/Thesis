import time
from Dataset_FM import get_dataloader_FM
from utils import print_and_write, reverse_transforms
from FunMatrix import EpipolarGeometry, update_epoch_stats
from FMatrixRegressor import FMatrixRegressor
from Dataset import get_data_loaders
from params import device, norm_mean, norm_std

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

def vis_trained(plots_path):
    model = FMatrixRegressor(lr_vit=2e-5, lr_mlp=2e-5, pretrained_path=plots_path)
    
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=1)
    for i, (img1, img2, label, pts1, pts2, seq_name) in enumerate(val_loader):
        img1, img2 = img1.to(device), img2.to(device)
        output = model.forward(img1, img2)

        epipolar_geo = EpipolarGeometry(img1[0], img2[0], output[0].detach(), pts1=pts1[0], pts2=pts2[0])
        epipolar_geo.visualize(idx=i, epipolar_lines_path=os.path.join("predicted_RealEstate", seq_name[0]))

def sed_gt():
    train_loader, val_loader, test_loader = get_data_loaders(train_size=3, batch_size=2)

    epoch_stats = {"test_algebraic_pred": torch.tensor(0), "test_algebraic_sqr_pred": torch.tensor(0), "test_RE1_pred": torch.tensor(0), "test_SED_pred": torch.tensor(0),
                   "test_algebraic_truth": torch.tensor(0), "test_algebraic_sqr_truth": torch.tensor(0), "test_RE1_truth": torch.tensor(0), "test_SED_truth": torch.tensor(0),
                   "test_loss": torch.tensor(0), "test_labels": torch.tensor([]), "test_outputs": torch.tensor([])}
    
    for i, (img1, img2, label, pts1, pts2, seq_name) in enumerate(test_loader):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        update_epoch_stats(epoch_stats, img1.detach(), img2.detach(), label.detach(), label.detach(), pts1, pts2, seq_name, data_type="test")
        if i==0: break

    print(f"""SED distance: {epoch_stats["test_SED_pred"]/i}
Algebraic distance: {epoch_stats["test_algebraic_pred"]/i}
RE1 distance: {epoch_stats["test_RE1_pred"]/i}""")


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

def sed_distance_trained(plots_path):
    model = FMatrixRegressor(lr_vit=2e-5, lr_mlp=2e-5, pretrained_path=plots_path)
    train_loader, val_loader, test_loader = get_data_loaders()

    epoch_stats = {"algebraic_pred": torch.tensor(0), "algebraic_sqr_pred": torch.tensor(0), "RE1_pred": torch.tensor(0), "SED_pred": torch.tensor(0), 
                "val_algebraic_pred": torch.tensor(0), "val_algebraic_sqr_pred": torch.tensor(0), "val_RE1_pred": torch.tensor(0), "val_SED_pred": torch.tensor(0), 
                "test_algebraic_pred": torch.tensor(0), "test_algebraic_sqr_pred": torch.tensor(0), "test_RE1_pred": torch.tensor(0), "test_SED_pred": torch.tensor(0),
                "algebraic_truth": torch.tensor(0), "algebraic_sqr_truth": torch.tensor(0), "RE1_truth": torch.tensor(0), "SED_truth": torch.tensor(0), 
                "val_algebraic_truth": torch.tensor(0), "val_algebraic_sqr_truth": torch.tensor(0), "val_RE1_truth": torch.tensor(0), "val_SED_truth": torch.tensor(0), 
                "test_algebraic_truth": torch.tensor(0), "test_algebraic_sqr_truth": torch.tensor(0), "test_RE1_truth": torch.tensor(0), "test_SED_truth": torch.tensor(0),
                "loss": torch.tensor(0), "val_loss": torch.tensor(0), "test_loss": torch.tensor(0), 
                "labels": torch.tensor([]), "outputs": torch.tensor([]), "val_labels": torch.tensor([]), "val_outputs": torch.tensor([]), "test_labels": torch.tensor([]), "test_outputs": torch.tensor([]),
                "file_num": 0}
    
    for i, (img1, img2, F, _) in enumerate(test_loader):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        output = model.forward(img1, img2)

        update_epoch_stats(epoch_stats, img1.detach(), img2.detach(), label.detach(), output, plots_path, data_type="test")

        if i == 10: break
    

    print(f"""SED distance: {epoch_stats["test_SED_pred"]/i}
    Algebraic distance: {epoch_stats["test_algebraic_pred"]/i}
    RE1 distance: {epoch_stats["test_RE1_pred"]/i}
    SED distance truth: {epoch_stats["test_SED_truth"]/i}
    Algebraic distance truth: {epoch_stats["test_algebraic_truth"]/i}
    RE1 distance truth: {epoch_stats["test_RE1_truth"]}"""/i)


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
    # Frozen: 0
    # mean_alg_0 = [0.2723333, 0.4836667, 0.3636667, 0.5166667, 0.5033333]
    # std_alg_0 = [0.00757, 0.0645, 0.03496, 0.03512, 0.05774]

    # mean_sed_0 = [0.275, 0.6866667, 0.44, 0.7676667, 0.7056667]
    # std_sed_0 = [0.00889, 0.16623, 0.0781, 0.11492, 0.05669]

    # mean_re1_0 = [0.079, 0.2536667, 0.1486667, 0.302, 0.2866667]
    # std_re1_0 = [0.00346, 0.0862, 0.02873, 0.00436, 0.05069]

    # # Frozen: 4
    # mean_alg_4 = [0.26867, 0.415, 0.39333, 0.52967, 0.59533]
    # std_alg_4 = [0.0255, 0.06366, 0.11676, 0.04409, 0.04324]

    # mean_sed_4 = [0.27933, 0.51633, 0.535, 0.749, 0.79167]
    # std_sed_4 = [0.04347, 0.11868, 0.25372, 0.07171, 0.11389]

    # mean_re1_4 = [0.08233, 0.187, 0.17533, 0.31433, 0.38933]
    # std_re1_4 = [0.01405, 0.05444, 0.08213, 0.04267, 0.04895]

    # # Frozen: 8
    # mean_alg_8 = [0.25733, 0.50167, 0.35333, 0.564, 0.54567]
    # std_alg_8 = [0.0181475, 0.0889288, 0.0404145, 0.0390512, 0.04782]

    # mean_sed_8 = [0.26733, 0.71433, 0.429, 0.813, 0.79167]
    # std_sed_8 = [0.0387599, 0.1795671, 0.0878806, 0.0713933, 0.11389]

    # mean_re1_8 = [0.07767, 0.294, 0.14867, 0.36367, 0.37433]
    # std_re1_8 = [0.0150444, 0.0930161, 0.0358515, 0.0119304, 0.089845]


    # PRETAINED VIT #
    # Frozen: 0
    # mean_alg_0 = [0.272, 0.49, 0.53333, 0.56667, 0.61333]
    # std_alg_0 = [0.02078, 0.04359, 0.04726, 0.15822, 0.07572]

    # mean_sed_0 = [0.27067, 0.71667, 0.78333, 0.88667, 0.89]
    # std_sed_0 = [0.02572, 0.10599, 0.10066, 0.28919, 0.10536]

    # mean_re1_0 = [0.07667, 0.28, 0.331, 0.39667, 0.41667]
    # std_re1_0 = [0.01155, 0.06083, 0.03568, 0.1601, 0.08021]

    # # Frozen: 4
    # mean_alg_4 = [0.27233, 0.45667, 0.53433, 0.50333, 0.56]
    # std_alg_4 = [0.03219, 0.0611, 0.03502, 0.10017, 0.06083]

    # mean_sed_4 = [0.28667, 0.64567, 0.76, 0.79667, 0.82]
    # std_sed_4 = [0.04509, 0.1328, 0.03606, 0.12014, 0.15524]

    # mean_re1_4 = [0.08467, 0.26, 0.33667, 0.34733, 0.39]
    # std_re1_4 = [0.01332, 0.08185, 0.02082, 0.07128, 0.1253]


    # RESNET #
    mean_alg_0 = [0.3, 0.41333, 0.52667, 0.57333, 0.62]
    std_alg_0 = [0.1852026, 0.0152753, 0.0981495, 0.0378594, 0.1493318]

    mean_sed_0 = [0.64333, 0.52, 0.72667, 0.79333, 0.86333]
    std_sed_0 = [0.2990541, 0.034641, 0.1850225, 0.065833, 0.2702468]

    mean_re1_0 = [0.14633, 0.22, 0.32, 0.35767, 0.39467]
    std_re1_0 = [0.1605937, 0.0264575, 0.1153256, 0.0292632, 0.1293265]

    # New X-axis with an extra point
    x_indices = range(len(mean_alg_0))  # For Frozen 0 (has an extra point)

    # Setting the X-axis labels to be flexible based on data points
    xticks_labels = ['2166', '1082', '540', '405', '269']  # 5 points for Frozen 0

    # Plotting the Frozen 0 run with solid lines (with extra point)
    plt.errorbar(x_indices, mean_alg_0, yerr=std_alg_0, marker='o', color='blue', linestyle='-', label='alg Frozen 0', capsize=5)
    plt.errorbar(x_indices, mean_sed_0, yerr=std_sed_0, marker='o', color='green', linestyle='-', label='SED Frozen 0', capsize=5)
    plt.errorbar(x_indices, mean_re1_0, yerr=std_re1_0, marker='o', color='orange', linestyle='-', label='RE1 Frozen 0', capsize=5)
    
    # Plotting the Frozen 4 run with dotted lines (ends earlier, 4 points)
    # plt.errorbar(x_indices, mean_alg_4, yerr=std_alg_4, marker='o', color='blue', linestyle=':', label='alg Frozen 4', capsize=5)
    # plt.errorbar(x_indices, mean_sed_4, yerr=std_sed_4, marker='o', color='green', linestyle=':', label='SED Frozen 4', capsize=5)
    # plt.errorbar(x_indices, mean_re1_4, yerr=std_re1_4, marker='o', color='orange', linestyle=':', label='RE1 Frozen 4', capsize=5)

    # Plotting the Frozen 8 run with dashed lines (ends earlier, 4 points)
    # plt.errorbar(x_indices, mean_alg_8, yerr=std_alg_8, marker='o', color='blue', linestyle='--', label='alg Frozen 8', capsize=5)
    # plt.errorbar(x_indices, mean_sed_8, yerr=std_sed_8, marker='o', color='green', linestyle='--', label='SED Frozen 8', capsize=5)
    # plt.errorbar(x_indices, mean_re1_8, yerr=std_re1_8, marker='o', color='orange', linestyle='--', label='RE1 Frozen 8', capsize=5)

    # Setting plot details
    plt.title('Mean Values with STD for frozen layers 0,4,8 with RESNET')
    plt.xlabel('Data Points')
    plt.ylabel('Mean Value ± STD')
    plt.xticks(range(len(xticks_labels)), labels=xticks_labels)  # Adjusting X-axis labels for Frozen 0
    plt.legend()
    plt.grid(True)

    # Show the combined plot
    plt.show()

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


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    plot_errors()