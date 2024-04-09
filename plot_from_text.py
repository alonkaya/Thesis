import re
import matplotlib.pyplot as plt
import numpy as np
# Initialize lists to hold parameter values
epochs = []
training_losses = []
training_maes = []
last_svs = []
alg_dists_truth = []
alg_dists_pred = []
alg_dists_pred_unorm = []
sed_dists_truth = []
sed_dists_pred = []
sed_dists_pred_unorm = []
# Open the file and process each line
with open("g.txt", 'r') as file:
    for line in file:
        # Extract epoch number
        epoch_match = re.search(r'Epoch (\d+)', line)
        if epoch_match:
            epochs.append(int(epoch_match.group(1)))

        # Extract training metrics
        if 'Training Loss' in line:
            training_loss_match = re.search(r'Training Loss: ([\d.]+)', line)
            training_mae_match = re.search(r'Training MAE: ([\d.]+)', line)
            last_sv_match = re.search(r'last sv: ([\d.e-]+)', line)
            if training_loss_match:
                training_losses.append(float(training_loss_match.group(1)))
            if training_mae_match:
                training_maes.append(float(training_mae_match.group(1)))
            if last_sv_match:
                last_svs.append(float(last_sv_match.group(1)))

        # Extract algebraic and SED distances
        alg_dist_match = re.search(r'algebraic dist truth: ([\d.e-]+), algebraic dist pred: ([\d.e-]+), algebraic dist pred unormalized: ([\d.e-]+)', line)
        sed_dist_match = re.search(r'SED dist truth: ([\d.e-]+), SED dist pred: ([\d.e-]+), SED dist pred unormalized: ([\d.e-]+)', line)
        if alg_dist_match:
            alg_dists_truth.append(float(alg_dist_match.group(1)))
            alg_dists_pred.append(float(alg_dist_match.group(2)))
            alg_dists_pred_unorm.append(float(alg_dist_match.group(3)))
        if sed_dist_match:
            sed_dists_truth.append(float(sed_dist_match.group(1)))
            sed_dists_pred.append(float(sed_dist_match.group(2)))
            sed_dists_pred_unorm.append(float(sed_dist_match.group(3)))

# Plotting function for each parameter
def plot_parameter(epochs, values, title, ylabel):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, values)  # Removed marker argument to display only the line
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)

    for i in range(0, len(epochs), max(1, len(epochs)//10)):
        plt.text(epochs[i], values[i], f'{values[i]:.4f}', fontsize=9, color='blue', ha='center', va='bottom')

    # Set the number of y-axis ticks to 10
    y_ticks = np.linspace(min(values), max(values), 10)
    plt.yticks(y_ticks)

    plt.grid(True)
    plt.show()

# Plotting graphs for each parameter
plot_parameter(epochs, training_losses, 'Training Loss per Epoch', 'Training Loss')
plot_parameter(epochs, training_maes, 'Training MAE per Epoch', 'Training MAE')
plot_parameter(epochs, last_svs, 'Last SV per Epoch', 'Last SV')
plot_parameter(epochs, alg_dists_truth, 'Algebraic Distance Truth per Epoch', 'Algebraic Distance Truth')
plot_parameter(epochs, alg_dists_pred, 'Algebraic Distance Predicted per Epoch', 'Algebraic Distance Predicted')
plot_parameter(epochs, alg_dists_pred_unorm, 'Algebraic Distance Predicted Unnormalized per Epoch', 'Algebraic Distance Pred Unnormalized')
plot_parameter(epochs, sed_dists_truth, 'SED Distance Truth per Epoch', 'SED Distance Truth')
plot_parameter(epochs, sed_dists_pred, 'SED Distance Predicted per Epoch', 'SED Distance Predicted')
plot_parameter(epochs, sed_dists_pred_unorm, 'SED Distance Predicted Unnormalized per Epoch', 'SED Distance Pred Unnormalized')
