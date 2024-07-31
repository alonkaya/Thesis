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
alg_sqr_dists = []
val_alg_sqr_dists = []
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
            val_mae_match = re.search(r'Val MAE: ([\d.]+)', line, re.IGNORECASE)
            if training_mae_match:
                training_maes.append(float(training_mae_match.group(1)))
            if val_mae_match:
                val_maes.append(float(val_mae_match.group(1)))

            # Extract and append the algebraic distances
            alg_dist_match = re.search(r'algebraic dist: ([\d.]+)', line, re.IGNORECASE)
            val_alg_dist_match = re.search(r'Val algebraic dist: ([\d.]+)', line, re.IGNORECASE)
            if alg_dist_match:
                alg_dists.append(float(alg_dist_match.group(1)))
            if val_alg_dist_match:
                val_alg_dists.append(float(val_alg_dist_match.group(1)))
                
            # Extract and append the algebraic sqr distances
            alg_sqr_dist_match = re.search(r'Algebraic sqr dist: ([\d.]+)', line, re.IGNORECASE)
            val_alg_sqr_dist_match = re.search(r'Val Algebraic sqr dist: ([\d.]+)', line, re.IGNORECASE)
            if alg_sqr_dist_match:
                alg_sqr_dists.append(float(alg_sqr_dist_match.group(1)))
            if val_alg_sqr_dist_match:
                val_alg_sqr_dists.append(float(val_alg_sqr_dist_match.group(1)))                

            # Extract and append the RE1 distances
            re1_dist_match = re.search(r'RE1 dist: ([\d.]+)', line)
            val_re1_dist_match = re.search(r'Val RE1 dist: ([\d.]+)', line, re.IGNORECASE)
            if re1_dist_match:
                re1_dists.append(float(re1_dist_match.group(1)))
            if val_re1_dist_match:
                val_re1_dists.append(float(val_re1_dist_match.group(1)))

            # Extract and append the SED distances
            sed_dist_match = re.search(r'SED dist: ([\d.]+)', line)
            val_sed_dist_match = re.search(r'Val SED dist: ([\d.]+)', line, re.IGNORECASE)
            if sed_dist_match:
                sed_dists.append(float(sed_dist_match.group(1)))
            if val_sed_dist_match:
                val_sed_dists.append(float(val_sed_dist_match.group(1)))

    return epochs, training_losses, val_losses, training_maes, val_maes, alg_dists, val_alg_dists, re1_dists, val_re1_dists, sed_dists, val_sed_dists, alg_sqr_dists, val_alg_sqr_dists

# Plotting function for each parameter
def plot_parameter(x, y1, y2, title, plots_path=None, x_label="Epochs", save=False):
    sliced = ""
    # if len(y1) > 3 and (y1[0] > y1[3] + 2000 or y2[0] > y2[3] + 2000):
    y1 = y1[5:]
    y2 = y2[5:]
    x = x[5:]
    sliced = " sliced"
    fig, axs = plt.subplots(1, 2, figsize=(18, 7))  # 1 row, 2 columns
    
    for ax, y_scale in zip(axs, ['linear', 'log']):
        ax.plot(x, y1, color='steelblue', label="Train")
        if y2 and len(y2)>0: ax.plot(x, y2, color='salmon', label="Test") 

        for i in range(0, len(y1), max(1, len(y1)//10)):
            ax.text(x[i], y1[i], f'{y1[i]:.3g}', fontsize=8, color='blue', ha='center', va='bottom')
            if y2: ax.text(x[i], y2[i], f'{y2[i]:.3g}', fontsize=8, color='red', ha='center', va='top')

        ax.set_xlabel(x_label)
        ax.set_ylabel(title if y_scale == 'linear' else f'{title} log scale')
        ax.set_title(f'{title} {y_scale} scale{sliced}')
    
        ax.set_yscale(y_scale)
        ax.grid(True)
        ax.legend()
    
    if save:
        os.makedirs(plots_path, exist_ok=True)
        plt.savefig(f"""{plots_path}/{title}.png""")  # Specify the filename and extension
    else:
        plt.show()


if __name__ == "__main__":
    plots_path = "plots/Stereo/Winners/SED_0.5__L2_1__huber_1__auged__lr_0.0001__conv__CLIP__use_reconstruction_True__BS_32__ratio_0.2__head_False"
    file_path = os.path.join(plots_path, "output.log")
    save = True

    process_epoch_stats(file_path)
    print(len(epochs), len(training_losses), len(val_losses), len(training_maes), len(val_maes), len(alg_dists), len(val_alg_dists), len(re1_dists), len(val_re1_dists), len(sed_dists), len(val_sed_dists), len(alg_sqr_dists), len(val_alg_sqr_dists))
    
    plot_parameter(epochs, training_losses, val_losses, "Loss", plots_path, save=save)
    plot_parameter(epochs, training_maes, val_maes, "MAE", plots_path, save=save)
    plot_parameter(epochs, alg_dists, val_alg_dists, "Algebraic Distance", plots_path, save=save)
    plot_parameter(epochs, re1_dists, val_re1_dists, "RE1 Distance", plots_path, save=save)
    plot_parameter(epochs, sed_dists, val_sed_dists, "SED Distance", plots_path, save=save)

    try:
        plot_parameter(epochs, alg_sqr_dists, val_alg_sqr_dists, "Algebraic Sqr Distance", plots_path, save=save)
    except:
        print("No Algebraic Sqr Distance found")