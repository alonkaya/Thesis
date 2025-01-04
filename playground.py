import os
import matplotlib.pyplot as plt
from params import *
from AffineRegressor import AffineRegressor
from Dataset import get_dataloaders

def parse_data_from_file(file_path):
    with open(file_path, 'r') as f:
        file_content = f.read()

    epochs = []
    train_loss, val_loss = [], []
    train_mae_shift, val_mae_shift = [], []
    train_euclidean_shift, val_euclidean_shift = [], []
    train_mae_angle, val_mae_angle = [], []
    train_mse_angle, val_mse_angle = [], []

    for line in file_content.splitlines():
        line = line.strip()
        
        # Parse the epoch, training loss, and validation loss
        if line.startswith("Epoch"):
            parts = line.split()
            epoch_num = int(parts[1].split("/")[0])
            epochs.append(epoch_num)

            # Extracting the loss values
            train_loss_value = float(parts[4])
            val_loss_value = float(parts[7])
            train_loss.append(train_loss_value)
            val_loss.append(val_loss_value)

        # Parse MAE Shift
        elif "Training MAE Shift" in line:
            parts = line.split()
            train_mae_shift_value = float(parts[3])
            val_mae_shift_value = float(parts[7])
            train_mae_shift.append(train_mae_shift_value)
            val_mae_shift.append(val_mae_shift_value)

        # Parse Euclidean Shift
        elif "Training Euclidean Shift" in line:
            parts = line.split()
            train_euclidean_shift_value = float(parts[3])
            val_euclidean_shift_value = float(parts[7])
            train_euclidean_shift.append(train_euclidean_shift_value)
            val_euclidean_shift.append(val_euclidean_shift_value)

        # Parse MAE Angle
        elif "Training MAE Angle" in line:
            parts = line.split()
            train_mae_angle_value = float(parts[3])
            val_mae_angle_value = float(parts[7])
            train_mae_angle.append(train_mae_angle_value)
            val_mae_angle.append(val_mae_angle_value)

        # Parse MSE Angle
        elif "Training MSE Angle" in line:
            parts = line.split()
            train_mse_angle_value = float(parts[3])
            val_mse_angle_value = float(parts[7])
            train_mse_angle.append(train_mse_angle_value)
            val_mse_angle.append(val_mse_angle_value)

    return (epochs, train_loss, val_loss, train_mae_shift, val_mae_shift, train_euclidean_shift, val_euclidean_shift, 
            train_mae_angle, val_mae_angle, train_mse_angle, val_mse_angle)



# Function to add data labels at intervals to avoid overlapping
def add_labels(x, y, ax, color, interval=2):
    for i in range(0, len(x), interval):  # Only add label at certain intervals
        ax.text(x[i], y[i], f'{y[i]:.4f}', ha='right', va='bottom', fontsize=8, color=color)

# Updated Plotting function
def plot_training_stats(epochs, train_loss, val_loss, train_mae_shift, val_mae_shift, 
                        train_euclidean_shift, val_euclidean_shift, 
                        train_mae_angle, val_mae_angle, 
                        train_mse_angle, val_mse_angle):
    
    plt.figure(figsize=(18, 12))
    
    # Loss
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(epochs, train_loss, label='Train Loss', color='blue', marker='o')
    ax1.plot(epochs, val_loss, label='Val Loss', color='orange', marker='o')
    add_labels(epochs, train_loss, ax1, 'blue', interval=3)  # Adjust the interval for labels
    add_labels(epochs, val_loss, ax1, 'orange', interval=3)
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True)

    # MAE Shift
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, train_mae_shift, label='Train MAE Shift', color='blue', marker='o')
    ax2.plot(epochs, val_mae_shift, label='Val MAE Shift', color='orange', marker='o')
    add_labels(epochs, train_mae_shift, ax2, 'blue', interval=3)
    add_labels(epochs, val_mae_shift, ax2, 'orange', interval=3)
    ax2.set_ylabel('MAE Shift')
    ax2.set_title('MAE Shift')
    ax2.legend()
    ax2.grid(True)

    # Euclidean Shift
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(epochs, train_euclidean_shift, label='Train Euclidean Shift', color='blue', marker='o')
    ax3.plot(epochs, val_euclidean_shift, label='Val Euclidean Shift', color='orange', marker='o')
    add_labels(epochs, train_euclidean_shift, ax3, 'blue', interval=3)
    add_labels(epochs, val_euclidean_shift, ax3, 'orange', interval=3)
    ax3.set_ylabel('Euclidean Shift')
    ax3.set_title('Euclidean Shift')
    ax3.legend()
    ax3.grid(True)

    # MAE Angle
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(epochs, train_mae_angle, label='Train MAE Angle', color='blue', marker='o')
    ax4.plot(epochs, val_mae_angle, label='Val MAE Angle', color='orange', marker='o')
    add_labels(epochs, train_mae_angle, ax4, 'blue', interval=3)
    add_labels(epochs, val_mae_angle, ax4, 'orange', interval=3)
    ax4.set_ylabel('MAE Angle')
    ax4.set_title('MAE Angle')
    ax4.legend()
    ax4.grid(True)

    # MSE Angle (Linear Scale)
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(epochs, train_mse_angle, label='Train MSE Angle', color='blue', marker='o')
    ax5.plot(epochs, val_mse_angle, label='Val MSE Angle', color='orange', marker='o')
    add_labels(epochs, train_mse_angle, ax5, 'blue', interval=3)
    add_labels(epochs, val_mse_angle, ax5, 'orange', interval=3)
    ax5.set_ylabel('MSE Angle')
    ax5.set_title('MSE Angle')
    ax5.legend()
    ax5.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_stats():
    file_path = "C:/Users/User/OneDrive/Thesis/Thesis/plots/BS_32__lr_6e-05__train_size_6144__model_CLIP__avg__alpha_10__angle_shift/output.log"

    epochs, train_loss, val_loss, train_mae_shift, val_mae_shift, train_euclidean_shift, val_euclidean_shift, \
    train_mae_angle, val_mae_angle, train_mse_angle, val_mse_angle = parse_data_from_file(file_path)
    plot_training_stats(epochs, train_loss, val_loss, train_mae_shift, val_mae_shift, train_euclidean_shift, val_euclidean_shift, train_mae_angle, val_mae_angle, train_mse_angle, val_mse_angle)

def test():
    pretrained_path = "plots/Affine/Ablation/BS_32__lr_6e-05__train_size_2048__CLIP__alpha_10__conv__original_rotated"

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE[0], train_length=64, val_length=val_length, test_length=64)

    model = AffineRegressor(LR[0], BATCH_SIZE[0], ALPHA[0], model_name=MODEL, avg_embeddings=AVG_EMBEDDINGS, plots_path="plots/test", pretrained_path=pretrained_path, use_conv=USE_CONV, num_epochs=NUM_EPOCHS)

    model.test(test_loader)

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    test()
    # plot_stats()