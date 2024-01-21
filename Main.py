from FMatrixRegressor import FMatrixRegressor
from params import *
from Dataset import train_loader, val_loader
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = FMatrixRegressor(mlp_hidden_sizes, num_output, pretrained_model_name=clip_model_name, lr=learning_rate, device=device, freeze_pretrained_model=False)
model = model.to(device)

print(f'learning_rate: {learning_rate}, mlp_hidden_sizes: {mlp_hidden_sizes}, jump_frames: {jump_frames},  add_penalty_loss: {add_penalty_loss}')
model.train_model(train_loader, val_loader, num_epochs=num_epochs)



