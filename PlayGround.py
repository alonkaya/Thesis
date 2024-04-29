from Dataset import get_data_loaders
from DatasetOneSequence import data_with_one_sequence
from FunMatrix import EpipolarGeometry
import numpy as np
import matplotlib.pyplot as plt
from params import norm_mean, norm_std
import os

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

    for i, (first_image, second_image, label, idx, sequence_path) in enumerate(val_loader):
        
        sequence_path = os.path.split(sequence_path[0])[0]
        epipolar_geo = EpipolarGeometry(first_image[0], second_image[0], F=label[0])
        epipolar_geo.visualize(idx=idx.item(), sequence_path=sequence_path, move_bad_images=True)

    for i, (first_image, second_image, label, idx, sequence_path) in enumerate(train_loader):
        sequence_path = os.path.split(sequence_path[0])[0]
        epipolar_geo = EpipolarGeometry(first_image[0], second_image[0], F=label[0])
        epipolar_geo.visualize(idx=idx.item(), sequence_path=sequence_path, move_bad_images=True)

if __name__ == "__main__":
    # Get the train_loader
    move_bad_images()
    # total_sed = 0
    # for j,p in enumerate(os.listdir("RealEstate10K/val_images")):
    #     train_loader, val_loader = data_with_one_sequence(1, sequence_name=p)
    #     sed = 0
    #     for i,(img1, img2, label) in enumerate(val_loader):
    #         epipolar_geo_pred = EpipolarGeometry(img1[0],img2[0], label[0]) 
    #         sed += epipolar_geo_pred.get_SED_distance()
    #     sed /= i
    #     print(f'Sequence name: {p}, sed: {sed}')

    
    # Iterate over the train_loader
    # for first_image, second_image, _ in train_loader:
    #     show_images(first_image, second_image)
        
    #     # Break or wait for user input to continue showing images
    #     input("Press Enter to continue...")  # Wait for user input to continue
    #     # If you want to break after the first batch, uncomment the following line
    #     # break