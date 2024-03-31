import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import os
import faulthandler

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
faulthandler.enable()

from utils import print_and_write, init_main
from FMatrixRegressor import FMatrixRegressor
from params import *
import itertools
from Dataset import * 

if __name__ == "__main__":
    init_main()

    # Iterate over each combination
    param_combinations = itertools.product(penalty_coeffs, penaltize_normalized_options, learning_rates_vit, learning_rates_mlp)
    
    for i, (penalty_coeff, penaltize_normalized, lr_vit, lr_mlp) in enumerate(param_combinations):
        model = FMatrixRegressor(lr_vit=lr_vit, 
                                 lr_mlp=lr_mlp,
                                 penalty_coeff=penalty_coeff,
                                 penaltize_normalized=penaltize_normalized, 
                                ).to(device)

        train_loader, val_loader = data_with_one_sequence(BATCH_SIZE, CUSTOMDATASET_TYPE)

        model.train_model(train_loader, val_loader, num_epochs=NUM_EPOCHS)




