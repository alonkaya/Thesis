# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

from utils import print_and_write, init_main
from FMatrixRegressor import FMatrixRegressor
from params import *
import itertools
from DatasetOneSequence import * 


if __name__ == "__main__":
    init_main()

    # Iterate over each combination
    param_combinations = itertools.product(learning_rates_vit, learning_rates_mlp)
    
    for i, (lr_vit, lr_mlp) in enumerate(param_combinations):
        model = FMatrixRegressor(lr_vit=lr_vit, 
                                 lr_mlp=lr_mlp,
                                ).to(device)

        train_loader, val_loader = data_with_one_sequence(BATCH_SIZE, CUSTOMDATASET_TYPE)
        
        parameters = f"""learning rate vit: {lr_vit}, learning rate mlp: {lr_mlp}, mlp_hidden_sizes: {MLP_HIDDEN_DIM}, jump_frames: {JUMP_FRAMES}, use_reconstruction_layer: {USE_RECONSTRUCTION_LAYER}
batch_size: {BATCH_SIZE}, train_seqeunces: {train_seqeunces}, val_sequences: {val_sequences}, RealEstate: {USE_REALESTATE}, batchnorm & dropout: {BN_AND_DO}, 
average embeddings: {AVG_EMBEDDINGS}, customdataset type: {CUSTOMDATASET_TYPE}, model: {MODEL}, augmentation: {AUGMENTATION}, enforce_rank_2:{ENFORCE_RANK_2}, predict pose: {PREDICT_POSE}, 
SVD coeff: {SVD_COEFF}, RE1 coeff: {RE1_COEFF} SED coeff: {SED_COEFF}, unforzen layers: {UNFROZEN_LAYERS}, group conv: {GROUP_CONV}\n\n"""
        print_and_write(parameters)

        model.train_model(train_loader, val_loader, num_epochs=NUM_EPOCHS)




