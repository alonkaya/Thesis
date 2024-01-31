from FMatrixRegressor import FMatrixRegressor
from params import *
from Dataset import get_data_loaders
import torch.multiprocessing as mp

if __name__ == "__main__":

    mp.set_start_method('spawn', force=True)

    model = FMatrixRegressor(mlp_hidden_sizes, num_output, pretrained_model_name=clip_model_name,
                            lr=learning_rate, freeze_pretrained_model=False,device=device)
    model = model.to(device)


    train_loader, val_loader = get_data_loaders()

    print(f'learning_rate: {learning_rate}, mlp_hidden_sizes: {mlp_hidden_sizes}, jump_frames: {jump_frames}, batch_size: {batch_size}')
    model.train_model(train_loader, val_loader, num_epochs=num_epochs)