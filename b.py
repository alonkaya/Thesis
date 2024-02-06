import matplotlib.pyplot as plt
import numpy as np

# Example data for demonstration
num_epochs = 15
all_train_loss = [1000,3,2,1,0.5,0.4,0.4,0.4,0.3,0.2,0.1,0.1,0.2,0.1,0.1]
all_val_loss = np.random.rand(num_epochs) * 0.2 + 0.1  # Simulated validation loss values
penalty_coeff, batch_size, learning_rate = 1,1, 2

def plot_over_epoch(x, y1, y2, title, penalty_coeff, batch_size, learning_rate, x_label="Epochs", show=True):

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns
    
    for ax, y_scale in zip(axs, ['linear', 'log']):
        ax.plot(x, y1, color='blue', label="Train")
        ax.plot(x, y2, color='orange', label="Val")
        
        for i, txt in enumerate(y1):
            ax.text(x[i], y1[i], f'{txt:.2f}', fontsize=8, color='blue', ha='center', va='bottom')

        # Annotate each point on the Val line
        for i, txt in enumerate(y2):
            ax.text(x[i], y2[i], f'{txt:.2f}', fontsize=8, color='orange', ha='center', va='top')

        ax.set_xlabel(x_label)
        ax.set_ylabel(title if y_scale == 'linear' else f'{title} log scale')
        ax.set_title(f'{title} - coeff: {penalty_coeff}, batch size: {batch_size}, lr: {learning_rate}, scale: {y_scale}')
    
        ax.set_yscale(y_scale)
        ax.set_xticks(x)
        ax.grid(True)
        ax.legend()

    plt.savefig(f'plots/{title} coeff {penalty_coeff} batch size {batch_size} lr {learning_rate}.png')  # Specify the filename and extension
    if show:
        plt.show()

# Using the function to plot the example data
plot_over_epoch(
    x=range(1, num_epochs + 1),
    y1=all_train_loss,
    y2=all_val_loss,
    title="Loss",
    penalty_coeff=penalty_coeff, 
    batch_size=batch_size, 
    learning_rate=learning_rate
)