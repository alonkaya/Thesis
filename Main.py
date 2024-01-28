from FMatrixRegressor import FMatrixRegressor
from params import *
from Dataset import get_data_loaders

model = FMatrixRegressor(mlp_hidden_sizes, num_output, pretrained_model_name=clip_model_name,
                         lr=learning_rate, freeze_pretrained_model=False)
model = model.to(device)

train_loader, val_loader = get_data_loaders()

print(f'learning_rate: {learning_rate}, mlp_hidden_sizes: {mlp_hidden_sizes}, jump_frames: {jump_frames}, use_reconstruction_layer: {use_reconstruction_layer}, batch_size: {batch_size}')
model.train_model(train_loader, val_loader, num_epochs=num_epochs)
