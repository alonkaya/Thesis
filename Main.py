from FMatrixRegressor import FMatrixRegressor
from params import *
from utils import print_and_write
from Dataset import get_data_loaders
import torch.multiprocessing as mp
import itertools
import os

if __name__ == "__main__":
    # mp.set_start_method('spawn', force=True)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print_and_write("###########################################################################################################\n\n")
    
    # Iterate over each combination
    param_combinations = itertools.product(batch_sizes, penalty_coeffs, penaltize_normalized_options, BATCHNORM_AND_DROPOUT_OPTIONS, learning_rates_vit, learning_rates_mlp)
    
    for i, (batch_size, penalty_coeff, penaltize_normalized, do_bn_and_do, lr_vit, lr_mlp) in enumerate(param_combinations):
        # if i==0: continue
        do_bn_and_do = do_bn_and_do if batch_size > 1 else False
        model = FMatrixRegressor(MLP_HIDDEN_DIM, NUM_OUTPUT, 
                                pretrained_model_name=CLIP_MODEL_NAME, lr_vit=lr_vit, lr_mlp=lr_mlp,
                                penalty_coeff=penalty_coeff, batch_size=batch_size, batchnorm_and_dropout=do_bn_and_do,
                                penaltize_normalized=penaltize_normalized, freeze_pretrained_model=False).to(device)

        train_loader, val_loader = get_data_loaders(batch_size)
        
        parameters = f"""learning rate vit: {lr_vit}, learning rate mlp: {lr_mlp}, mlp_hidden_sizes: {MLP_HIDDEN_DIM}, jump_frames: {JUMP_FRAMES}, penalty_coeff: {penalty_coeff}, use_reconstruction_layer: {USE_RECONSTRUCTION_LAYER}
batch_size: {batch_size}, train_seqeunces: {train_seqeunces}, val_sequences: {val_sequences}, penaltize_normalized: {penaltize_normalized}, RealEstate: {USE_REALESTATE} batchnorm & dropout: {do_bn_and_do}\n\n"""
        print_and_write(parameters)

        model.train_model(train_loader, val_loader, num_epochs=NUM_EPOCHS)
