from FMatrixRegressor import FMatrixRegressor
from params import *
from Dataset import get_data_loaders
import torch.multiprocessing as mp
import itertools
import os
if __name__ == "__main__":

    mp.set_start_method('spawn', force=True)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # Iterate over each combination
    param_combinations = itertools.product(batch_sizes, penalty_coeffs, penaltize_normalized_options)
    
    for batch_size, penalty_coeff, penaltize_normalized in param_combinations:
        model = FMatrixRegressor(mlp_hidden_sizes, num_output, 
                                pretrained_model_name=clip_model_name, lr=learning_rate, 
                                penalty_coeff=penalty_coeff, batch_size=batch_size, penaltize_normalized=penaltize_normalized, 
                                freeze_pretrained_model=False).to(device)

        train_loader, val_loader = get_data_loaders(batch_size)
        
        parameters = f'learning_rate: {learning_rate}, mlp_hidden_sizes: {mlp_hidden_sizes}, jump_frames: {jump_frames}, penalty_coeff: {penalty_coeff}, use_reconstruction_layer: {use_reconstruction_layer}, batch_size: {batch_size}, train_seqeunces: {train_seqeunces}, val_sequences: {val_sequences}, penaltize_normalized: {penaltize_normalized}\n\n'
        with open("output.txt", "a") as f:
            f.write(parameters)
            print(parameters)

        model.train_model(train_loader, val_loader, num_epochs=num_epochs)
