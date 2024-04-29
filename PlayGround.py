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




if __name__ == "__main__":
    # Get the train_loader
    train_loader, val_loader = data_with_one_sequence(1, sequence_name='bc0ebb7482f14795')
    for i,(img1, img2, label) in enumerate(val_loader):
        epipolar_geo_pred = EpipolarGeometry(img1[0],img2[0], label[0]) 
        epipolar_geo_pred.visualize(sqResultDir='predicted_epipole_lines_bc0ebb7482f14795', file_num=i)
    #     i+=1
    #     if epipolar_geo_pred.get_SED_distance() > thresh:
    #         sed+=1
    # print(sed, i)
    
    # Iterate over the train_loader
    # for first_image, second_image, _ in train_loader:
    #     show_images(first_image, second_image)
        
    #     # Break or wait for user input to continue showing images
    #     input("Press Enter to continue...")  # Wait for user input to continue
    #     # If you want to break after the first batch, uncomment the following line
    #     # break