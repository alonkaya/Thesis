from utils import plot_over_epoch
import matplotlib.pyplot as plt

# Function to parse the statistics from each line
def parse_line(first_line, second_line):
    first_line = first_line.split()
    epoch = int(first_line[1].split('/')[0])
    training_loss = float(first_line[4])
    val_loss = float(first_line[7])
    training_mae = float(first_line[10])
    val_mae = float(first_line[13])
    penalty = float(first_line[15])

    second_line = second_line.split()
    train_avg_epipolar_error = float(second_line[7])
    val_avg_epipolar_error = float(second_line[-1])

    return epoch, training_loss, val_loss, training_mae, val_mae, penalty, train_avg_epipolar_error, val_avg_epipolar_error

# Lists to store data for each variable
epochs = []
training_loss = []
val_loss = []
training_mae = []
val_mae = []
penalty = []
train_avg_epipolar_error = []
val_avg_epipolar_error = []

# Read data from file
with open('output.txt', 'r') as file:
    lines = file.readlines()
    for i in range(0, len(lines), 2):
        epoch, tr_loss, v_loss, tr_mae, v_mae, pen, tr_avg_epi, v_avg_epi = parse_line(lines[i], lines[i+1])        
        training_loss.append(tr_loss)
        val_loss.append(v_loss)
        training_mae.append(tr_mae)
        val_mae.append(v_mae)
        penalty.append(pen)
        train_avg_epipolar_error.append(tr_avg_epi)
        val_avg_epipolar_error.append(v_avg_epi)

if __name__ == "__main__":
    start_epoch = 3
    num_epochs = 100
    plot_over_epoch(x=range(start_epoch, num_epochs + 1), y=training_loss[start_epoch-1:],
                            x_label="Epoch", y_label='Training Loss', show=True)
    plot_over_epoch(x=range(start_epoch, num_epochs + 1), y=val_loss[start_epoch-1:],
                    x_label="Epoch", y_label='Validation Loss', show=True)
    plot_over_epoch(x=range(start_epoch, num_epochs + 1), y=training_mae[start_epoch-1:],
                    x_label="Epoch", y_label='Training MAE', show=True)
    plot_over_epoch(x=range(start_epoch, num_epochs + 1), y=val_mae[start_epoch-1:],
                    x_label="Epoch", y_label='VAlidation MAE', show=True)
    # plot_over_epoch(x=range(1, num_epochs + 1), y=ec_err_pred, x_label="Epoch", y_label='Training epipolar constraint err for pred F', show=show_plots)
    plot_over_epoch(x=range(start_epoch, num_epochs + 1), y=train_avg_epipolar_error[start_epoch-1:], 
                    x_label="Epoch", y_label='Train epipolar constraint err for pred F unormalized', show=True)
    # plot_over_epoch(x=range(1, num_epochs + 1), y=val_ec_err_pred, x_label="Epoch", y_label='Val epipolar constraint err for pred F', show=show_plots)
    plot_over_epoch(x=range(start_epoch, num_epochs + 1), y=val_avg_epipolar_error[start_epoch-1:], 
                    x_label="Epoch", y_label='Val epipolar constraint err for pred F unormalized', show=True)
    plot_over_epoch(x=range(start_epoch, num_epochs + 1), y=penalty[start_epoch-1:], 
                    x_label="Epoch", y_label='Additional loss penalty for last singular value', show=True)
