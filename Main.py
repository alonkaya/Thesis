from FMatrixRegressor import FMatrixRegressor
from params import *
from Dataset import get_data_loaders
import torch.multiprocessing as mp

if __name__ == "__main__":

    mp.set_start_method('spawn', force=True)

    model = FMatrixRegressor(mlp_hidden_sizes, num_output, pretrained_model_name=clip_model_name, lr=learning_rate, freeze_pretrained_model=False).to(device)
    
    for penalty_coeff, batch_size in zip(penalty_coeffs, batch_sizes):
        train_loader, val_loader = get_data_loaders(batch_size)

        print(f'learning_rate: {learning_rate}, mlp_hidden_sizes: {mlp_hidden_sizes}, jump_frames: {jump_frames}, penalty_coeff: {penalty_coeff}, use_reconstruction_layer: {use_reconstruction_layer}, batch_size: {batch_size}, train_seqeunces: {train_seqeunces}, val_sequences: {val_sequences}')

        model.train_model(train_loader, val_loader, num_epochs=num_epochs, penalty_coeff=penalty_coeff, batch_size=batch_size)
