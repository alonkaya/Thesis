import os
import re
import matplotlib.pyplot as plt
import numpy as np

epochs = []
training_losses = []
val_losses = []
training_maes = []
val_maes = []
alg_dists = []
val_alg_dists = []
re1_dists = []
val_re1_dists = []
sed_dists = []
val_sed_dists = []

def process_epoch_stats(file_path):
    # Open the file and iterate over each line
    with open(file_path, 'r') as file:
        for line in file:
            # Check if the line contains epoch information
            epoch_match = re.search(r'Epoch (\d+)/', line)
            if epoch_match:
                epochs.append(int(epoch_match.group(1)))

            # Extract and append the training and validation losses
            training_loss_match = re.search(r'Training Loss: ([\d.]+)', line)
            val_loss_match = re.search(r'Val Loss: ([\d.]+)', line)
            if training_loss_match:
                training_losses.append(float(training_loss_match.group(1)))
            if val_loss_match:
                val_losses.append(float(val_loss_match.group(1)))

            # Extract and append the training and validation MAEs
            training_mae_match = re.search(r'Training MAE: ([\d.]+)', line)
            val_mae_match = re.search(r'Val MAE: ([\d.]+)', line)
            if training_mae_match:
                training_maes.append(float(training_mae_match.group(1)))
            if val_mae_match:
                val_maes.append(float(val_mae_match.group(1)))

            # Extract and append the algebraic distances
            alg_dist_match = re.search(r'algebraic dist: ([\d.]+)', line)
            val_alg_dist_match = re.search(r'val algebraic dist: ([\d.]+)', line)
            if alg_dist_match:
                alg_dists.append(float(alg_dist_match.group(1)))
            if val_alg_dist_match:
                val_alg_dists.append(float(val_alg_dist_match.group(1)))

            # Extract and append the RE1 distances
            re1_dist_match = re.search(r'RE1 dist: ([\d.]+)', line)
            val_re1_dist_match = re.search(r'val RE1 dist: ([\d.]+)', line)
            if re1_dist_match:
                re1_dists.append(float(re1_dist_match.group(1)))
            if val_re1_dist_match:
                val_re1_dists.append(float(val_re1_dist_match.group(1)))

            # Extract and append the SED distances
            sed_dist_match = re.search(r'SED dist: ([\d.]+)', line)
            val_sed_dist_match = re.search(r'val SED dist: ([\d.]+)', line)
            if sed_dist_match:
                sed_dists.append(float(sed_dist_match.group(1)))
            if val_sed_dist_match:
                val_sed_dists.append(float(val_sed_dist_match.group(1)))


# Plotting function for each parameter
def plot_parameter(x, y1, y2, title, plots_path=None, x_label="Epochs"):
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))  # 1 row, 2 columns
    
    for ax, y_scale in zip(axs, ['linear', 'log']):
        ax.plot(x, y1, color='steelblue', label="Train")
        if y2 and len(y2)>0: ax.plot(x, y2, color='salmon', label="Test") 

        for i in range(0, len(y1), max(1, len(y1)//10)):
            ax.text(x[i], y1[i], f'{y1[i]:.4f}', fontsize=9, color='blue', ha='center', va='bottom')
            if y2: ax.text(x[i], y2[i], f'{y2[i]:.4f}', fontsize=9, color='red', ha='center', va='top')

        ax.set_xlabel(x_label)
        ax.set_ylabel(title if y_scale == 'linear' else f'{title} log scale')
        ax.set_title(f'{title} {y_scale} scale')
    
        ax.set_yscale(y_scale)
        ax.grid(True)
        ax.legend()

    os.makedirs(plots_path, exist_ok=True)
    plt.savefig(f"""{plots_path}/{title}.png""")  # Specify the filename and extension
    
    # plt.show()


if __name__ == "__main__":
    file_path = "plots/RealEstate/SVD_0__RE1_0__SED_0__ALG_0.1__lr_2e-05__avg_embeddings_True__model_CLIP__use_reconstruction_True__Augmentation_False__Conv_True/output.log"
    plots_path = "plots/RealEstate/SVD_0__RE1_0__SED_0__ALG_0.1__lr_2e-05__avg_embeddings_True__model_CLIP__use_reconstruction_True__Augmentation_False__Conv_True"
    process_epoch_stats(file_path)
    # plot_parameter(epochs, training_losses, val_losses, "Loss", plots_path)
    # plot_parameter(epochs, training_maes, val_maes, "MAE", plots_path)
    # plot_parameter(epochs, alg_dists, val_alg_dists, "Algebraic Distance", plots_path)
    # plot_parameter(epochs, re1_dists, val_re1_dists, "RE1 Distance", plots_path)
    plot_parameter(epochs, sed_dists, val_sed_dists, "SED Distance", plots_path)

    
