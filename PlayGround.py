import cv2
from Dataset import get_data_loaders
from FMatrixRegressor import FMatrixRegressor
from FunMatrix import EpipolarGeometry, update_epoch_stats
from params import device, norm_mean, norm_std
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import re
from utils import divide_by_dataloader, points_histogram, reverse_transforms

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
    train_loader, val_loader, test_loader = get_data_loaders()
    total_sed = 0
    for i, (img1, img2, label, pts1, pts2, seq_name) in enumerate(test_loader):
        
        # Convert grayscale tensors to numpy arrays for matplotlib
        img0_np = reverse_transforms(img1[0].cpu(), mean=norm_mean.cpu(), std=norm_std.cpu())  # shape (H, W, C)
        img1_np = reverse_transforms(img2[0].cpu(), mean=norm_mean.cpu(), std=norm_std.cpu())  # shape (H, W, C)

        # # Create a subplot with two images
        # fig, axs = plt.subplots(1, 2, figsize=(15, 7))

        # # Display the first image with keypoints
        # axs[0].imshow(img0_np, cmap='gray')
        # axs[0].scatter(pts1[:, 0].numpy(), pts1[:, 1].numpy(), c='r', s=10)
        # # axs[0].set_title(f'Image 0 - {idx}')
        # axs[0].axis('off')

        # # Display the second image with keypoints
        # axs[1].imshow(img1_np, cmap='gray')
        # axs[1].scatter(pts2[:, 0].numpy(), pts2[:, 1].numpy(), c='r', s=10)
        # # axs[1].set_title(f'Image 1 - {idx}')
        # axs[1].axis('off')

        # # Show the plot
        # # plt.show()
        # plt.savefig(f'gt_epiliines/{seq_name[0]}/gt_{i}.png')

        # Function to draw points on the image
        def draw_points(image, points, color=(0, 255, 0)):
            print(image.shape)
            for point in points:
                cv2.circle(image, (point[0], point[1]), 5, color, -1)

        # Draw points on the images
        draw_points(img0_np, pts1.cpu().numpy())
        draw_points(img1_np, pts2.cpu().numpy())

        # Concatenate images horizontally
        combined_image = np.hstack((img0_np, img1_np))

        # Save the combined image
        cv2.imwrite(f'gt_epilines/{seq_name[0]}/gt_{i}.png', combined_image)

        if i == 20: break
        
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


def sed_distance_gt():
    train_loader, val_loader, test_loader = get_data_loaders()

    epoch_stats = {"test_algebraic_pred": torch.tensor(0), "test_algebraic_sqr_pred": torch.tensor(0), "test_RE1_pred": torch.tensor(0), "test_SED_pred": torch.tensor(0),
                   "test_algebraic_truth": torch.tensor(0), "test_algebraic_sqr_truth": torch.tensor(0), "test_RE1_truth": torch.tensor(0), "test_SED_truth": torch.tensor(0),
                   "test_loss": torch.tensor(0), "test_labels": torch.tensor([]), "test_outputs": torch.tensor([])}
    
    for i, (img1, img2, label, pts1, pts2, _) in enumerate(test_loader):
        img1, img2, label, pts1, pts2 = img1.to(device), img2.to(device), label.to(device), pts1.to(device), pts2.to(device)

        update_epoch_stats(epoch_stats, img1.detach(), img2.detach(), label.detach(), label.detach(), pts1, pts2, "", data_type="test")
        print(i)
        if i == 50: break
    # divide_by_dataloader(epoch_stats, len_test_loader=len(test_loader))
    
    print(f'test_algebraic_pred: {epoch_stats["test_algebraic_pred"]/(i+1)}')
    print(f'test_algebraic_sqr_pred: {epoch_stats["test_algebraic_sqr_pred"]/(i+1)}')
    print(f'test_RE1_pred: {epoch_stats["test_RE1_pred"]/(i+1)}')
    print(f'test_SED_pred: {epoch_stats["test_SED_pred"]/(i+1)}')
    print()

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

from torchvision import transforms

if __name__ == "__main__":
    # file_path = "plots/Stereo/SED_0.05__lr_2e-05__avg_embeddings_True__conv_False__model_CLIP__use_reconstruction_True__Augment_True__rc_True"
    # update_epochs(file_path, 114)
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    vis_gt()

