import matplotlib.pyplot as plt
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_input, mlp_hidden_sizes, num_output):
        super(MLP, self).__init__()
        mlp_layers = []
        prev_size = num_input
        for hidden_size in mlp_hidden_sizes:
            mlp_layers.append(nn.Linear(prev_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            prev_size = hidden_size
        mlp_layers.append(nn.Linear(prev_size, num_output))

        self.layers = nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.layers(x)

def plot_over_epoch(x, y, x_label, y_label, connecting_lines=True):
    
    if connecting_lines:
      plt.plot(x, y)
    else:
      plt.plot(x, y, marker='o', linestyle='')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{y_label} over {x_label}')
    plt.show()