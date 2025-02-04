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


def mult_by_range(CLIP_Shift_Mean, CLIP_Shift_STD, CLIP_Angle_Mean, CLIP_Angle_STD, CLIP_16_Shift_Mean, CLIP_16_Shift_STD, CLIP_16_Angle_Mean, CLIP_16_Angle_STD, RESNET_Shift_Mean, RESNET_Shift_STD, RESNET_Angle_Mean, RESNET_Angle_STD, DINO_Shift_mean, DINO_Shift_std, DINO_Angle_mean, DINO_Angle_std, EFFICIENT_shift_mean, EFFICIENT_shift_std, EFFICIENT_angle_mean, EFFICIENT_angle_std):   
    CLIP_Shift_Mean = [x * SHIFT_RANGE for x in CLIP_Shift_Mean]
    CLIP_Shift_STD = [x * SHIFT_RANGE for x in CLIP_Shift_STD]
    CLIP_Angle_Mean = [x * ANGLE_RANGE for x in CLIP_Angle_Mean]
    CLIP_Angle_STD = [x * ANGLE_RANGE for x in CLIP_Angle_STD]

    CLIP_16_Shift_Mean = [x * SHIFT_RANGE for x in CLIP_16_Shift_Mean]
    CLIP_16_Shift_STD = [x * SHIFT_RANGE for x in CLIP_16_Shift_STD]
    CLIP_16_Angle_Mean = [x * ANGLE_RANGE for x in CLIP_16_Angle_Mean]
    CLIP_16_Angle_STD = [x * ANGLE_RANGE for x in CLIP_16_Angle_STD]

    RESNET_Shift_Mean = [x * SHIFT_RANGE for x in RESNET_Shift_Mean]
    RESNET_Shift_STD = [x * SHIFT_RANGE for x in RESNET_Shift_STD]
    RESNET_Angle_Mean = [x * ANGLE_RANGE for x in RESNET_Angle_Mean]
    RESNET_Angle_STD = [x * ANGLE_RANGE for x in RESNET_Angle_STD]

    DINO_Shift_mean = [x * SHIFT_RANGE for x in DINO_Shift_mean]
    DINO_Shift_std = [x * SHIFT_RANGE for x in DINO_Shift_std]
    DINO_Angle_mean = [x * ANGLE_RANGE for x in DINO_Angle_mean]
    DINO_Angle_std = [x * ANGLE_RANGE for x in DINO_Angle_std]

    EFFICIENT_shift_mean = [x * SHIFT_RANGE for x in EFFICIENT_shift_mean]
    EFFICIENT_shift_std = [x * SHIFT_RANGE for x in EFFICIENT_shift_std]
    EFFICIENT_angle_mean = [x * ANGLE_RANGE for x in EFFICIENT_angle_mean]
    EFFICIENT_angle_std = [x * ANGLE_RANGE for x in EFFICIENT_angle_std]

    return CLIP_Shift_Mean, CLIP_Shift_STD, CLIP_Angle_Mean, CLIP_Angle_STD, CLIP_16_Shift_Mean, CLIP_16_Shift_STD, CLIP_16_Angle_Mean, CLIP_16_Angle_STD, RESNET_Shift_Mean, RESNET_Shift_STD, RESNET_Angle_Mean, RESNET_Angle_STD, DINO_Shift_mean, DINO_Shift_std, DINO_Angle_mean, DINO_Angle_std, EFFICIENT_shift_mean, EFFICIENT_shift_std, EFFICIENT_angle_mean, EFFICIENT_angle_std 

def plot_results():
    CLIP_Shift_Mean = [0.023, 0.02835, 0.0445, 0.0673, 0.0735]
    CLIP_Shift_STD = [0.001414214, 0.000919239, 0.004949747, 0.006646804, 0.003535534]
    CLIP_Angle_Mean	= [0.0175,	0.0195,	0.0365,	0.0515,	0.0605]
    CLIP_Angle_STD = [0.000707107, 0.00212132, 0.004949747, 0.004949747, 0.007778175]

    CLIP_16_Shift_Mean =[0.0225,        0.0255,     0.04,       0.052,         0.0595]
    CLIP_16_Shift_STD = [0.000707107, 0.000707107, 0.004242641, 0.002828427,     0.00212132]
    CLIP_16_Angle_Mean =[0.0205,        0.0225, 	0.032,	    0.04075,         0.0445]
    CLIP_16_Angle_STD =	[0.003535534, 0.00212132,	0.004242641, 0.00106066,  0.004949747]

    RESNET_Shift_Mean = [0.0283,	   0.0353,	       0.0451,	     0.05675,	        0.067]
    RESNET_Shift_STD =	[0.000989949,  0.008061017,    0.002404163,  0.00106066,	0.002828427]
    RESNET_Angle_Mean =	[0.032,	        0.03,          0.04,         0.0465,         0.053]
    RESNET_Angle_STD =	[0.005656854,  0.006434672,    0.00212132,   0.006363961,	0.004242641]


    DINO_Shift_mean = [0.0193,	    0.0222,	      0.032,	    0.049,	      0.05835]
    DINO_Shift_std = [0.000707107,	0.000848528,  0.006505382,	0.005656854,  0.002333452]
    DINO_Angle_mean = [0.01565,	    0.01895,	  0.0267,	    0.03825,	  0.04285]
    DINO_Angle_std = [0.002616295,	0.004596194,  0.003252691,	0.005303301,  0.004454773]

    EFFICIENT_shift_mean = [0.02655,	    0.0285,	      0.0405,	    0.0482,	      0.0629]
    EFFICIENT_shift_std =  [0.000777817,    0.00212132,   0.000707107,	0.009616652,  0.001555635]
    EFFICIENT_angle_mean = [0.02365,	    0.0256,	      0.0343,	    0.045,	      0.048]
    EFFICIENT_angle_std =  [0.00516188,	0.004808326,  0.013152186,	0.007071068,  0.001414214]



    CLIP_Shift_Mean, CLIP_Shift_STD, CLIP_Angle_Mean, CLIP_Angle_STD, CLIP_16_Shift_Mean, CLIP_16_Shift_STD, CLIP_16_Angle_Mean, CLIP_16_Angle_STD, RESNET_Shift_Mean, RESNET_Shift_STD, RESNET_Angle_Mean, RESNET_Angle_STD, DINO_Shift_mean, DINO_Shift_std, DINO_Angle_mean, DINO_Angle_std, EFFICIENT_shift_mean, EFFICIENT_shift_std, EFFICIENT_angle_mean, EFFICIENT_angle_std  =  mult_by_range(CLIP_Shift_Mean, CLIP_Shift_STD, CLIP_Angle_Mean, CLIP_Angle_STD, CLIP_16_Shift_Mean, CLIP_16_Shift_STD, CLIP_16_Angle_Mean, CLIP_16_Angle_STD, RESNET_Shift_Mean, RESNET_Shift_STD, RESNET_Angle_Mean, RESNET_Angle_STD, DINO_Shift_mean, DINO_Shift_std, DINO_Angle_mean, DINO_Angle_std, EFFICIENT_shift_mean, EFFICIENT_shift_std, EFFICIENT_angle_mean, EFFICIENT_angle_std) 

    colors = [
        'xkcd:electric blue',
        'xkcd:red',
        'xkcd:bright purple', #magneta
        'xkcd:emerald',
        'xkcd:dark navy blue'
    ]    
    markers = ['o', 's', '^', 'D', 'x']  # Markers for each model
    linestyles = ['-', '--']  # Line styles for each model    
    capsize= 4      # Width of the error bars
    linewidth= 1.2  # Width of the line
    markersize= 4.4 # size of the dots
    
    os.makedirs('results', exist_ok=True)
    x_indices = range(len(RESNET_Shift_Mean))  # For Frozen 0 (has an extra point)
    xticks_labels = ['4048', '1024', '256', '64', '32']  # 5 points for Frozen 0

    fig1=plt.figure(1, figsize=(10, 6))
    plt.errorbar(x_indices, CLIP_Shift_Mean, yerr=CLIP_Shift_STD, label='CLIP-ViT/B32 Translation Error', color=colors[0], marker=markers[0], linestyle=linestyles[0], capsize=capsize, linewidth=linewidth, markersize=markersize)
    plt.errorbar(x_indices, CLIP_16_Shift_Mean, yerr=CLIP_16_Shift_STD, label='CLIP-ViT/B16 Translation Error', color=colors[1], marker=markers[1], linestyle=linestyles[0], capsize=capsize, linewidth=linewidth, markersize=markersize)
    plt.errorbar(x_indices, RESNET_Shift_Mean, yerr=RESNET_Shift_STD,label='ResNet-152 Translation Error', color=colors[2], marker=markers[2], linestyle=linestyles[1], capsize=capsize, linewidth=linewidth, markersize=markersize)
    plt.errorbar(x_indices, DINO_Shift_mean, yerr=DINO_Shift_std,  label='DINO-ViT/B16 Translation Error', color=colors[3], marker=markers[3], linestyle=linestyles[0], capsize=capsize, linewidth=linewidth, markersize=markersize)
    plt.errorbar(x_indices, EFFICIENT_shift_mean, yerr=EFFICIENT_shift_std, label='EfficientNet Translation Error', color=colors[4], marker=markers[4], linestyle=linestyles[1], capsize=capsize, linewidth=linewidth, markersize=markersize)
    plt.title('Translation estimation errors', fontsize=15)
    plt.xlabel('Number of training samples', fontsize=13) 
    plt.ylabel('Mean Values + STD', fontsize=13) 
    plt.xticks(range(len(xticks_labels)), labels=xticks_labels)  # Adjusting X-axis labels for Frozen 0
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)
    fig1.savefig('results/Affine_Translation.png')

    fig2=plt.figure(2, figsize=(10, 6))
    plt.errorbar(x_indices, CLIP_Angle_Mean, yerr=CLIP_Angle_STD, label='CLIP-ViT/B32 Rotation Error', color=colors[0], marker=markers[0], linestyle=linestyles[0], capsize=capsize, linewidth=linewidth, markersize=markersize)
    plt.errorbar(x_indices, CLIP_16_Angle_Mean, yerr=CLIP_16_Angle_STD,  label='CLIP-ViT/B16 Rotation Error', color=colors[1], marker=markers[1], linestyle=linestyles[0], capsize=capsize, linewidth=linewidth, markersize=markersize)
    plt.errorbar(x_indices, RESNET_Angle_Mean, yerr=RESNET_Angle_STD, label='ResNet Rotation Error', color=colors[2], marker=markers[2], linestyle=linestyles[1], capsize=capsize, linewidth=linewidth, markersize=markersize)
    plt.errorbar(x_indices, DINO_Angle_mean, yerr=DINO_Angle_std, label='DINO-ViT/B16 Rotation Error', color=colors[3], marker=markers[3], linestyle=linestyles[0], capsize=capsize, linewidth=linewidth, markersize=markersize)
    plt.errorbar(x_indices, EFFICIENT_angle_mean, yerr=EFFICIENT_angle_std, label='EfficientNet Rotation Error', color=colors[4], marker=markers[4], linestyle=linestyles[1], capsize=capsize, linewidth=linewidth, markersize=markersize)
    plt.title('Rotation estimation errors', fontsize=15)
    plt.xlabel('Number of training samples', fontsize=13) 
    plt.ylabel('Mean Values + STD', fontsize=13) 
    plt.xticks(range(len(xticks_labels)), labels=xticks_labels)  # Adjusting X-axis labels for Frozen 0
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)
    fig2.savefig('results/Affine_Rotation.png')


def test():
    prefix = '/mnt_hdd15tb/alonkay/Thesis/'
    pretrained_path = "plots/Affine/BS_4__lr_0.0001__alpha_10__conv__original_rotated_angle_30__shift_32/CLIP/size_64__frozen_0"
    pretrained_path = os.path.join(prefix, pretrained_path)

    batch_size=1
    
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size, train_length=64, val_length=val_length, test_length=test_length)

    model = AffineRegressor(LR[0], batch_size, ALPHA[0], model_name=MODEL, avg_embeddings=AVG_EMBEDDINGS, plots_path="plots/test", pretrained_path=pretrained_path, use_conv=USE_CONV, num_epochs=NUM_EPOCHS)

    print(model.start_epoch)
    model.test(test_loader)

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # test()
    # plot_stats()
    plot_results()