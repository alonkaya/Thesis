# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'

from Dataset import get_data_loaders
from utils import print_and_write, init_main
from FMatrixRegressor import FMatrixRegressor
from params import *
import itertools
from DatasetOneSequence import * 


if __name__ == "__main__":
    init_main()

    # Iterate over each combination
    param_combinations = itertools.product(learning_rates_vit, learning_rates_mlp, ALG_COEFF, RE1_COEFF, SED_COEFF)

    for i, (lr_vit, lr_mlp, alg_coeff, re1_coeff, sed_coeff) in enumerate(param_combinations):

        plots_path = os.path.join('plots', 'RealEstate', 
                          f"""SVD_{LAST_SV_COEFF}__RE1_{re1_coeff}__SED_{sed_coeff}__ALG_{alg_coeff}__lr_{learning_rates_mlp[0]}__\
avg_embeddings_{AVG_EMBEDDINGS}__model_{"CLIP" if MODEL == CLIP_MODEL_NAME else "Google ViT"}__\
predict_pose_{PREDICT_POSE}__use_reconstruction_{USE_RECONSTRUCTION_LAYER}__Augmentation_{AUGMENTATION}""")
        
        model = FMatrixRegressor(lr_vit=lr_vit, lr_mlp=lr_mlp, alg_coeff=alg_coeff, re1_coeff=re1_coeff, sed_coeff=sed_coeff, plots_path=plots_path).to(device)

        train_loader, val_loader = get_data_loaders(BATCH_SIZE)

        
        parameters = f"""learning rate vit: {lr_vit}, learning rate mlp: {lr_mlp}, mlp_hidden_sizes: {MLP_HIDDEN_DIM}, jump_frames: {JUMP_FRAMES}, use_reconstruction_layer: {USE_RECONSTRUCTION_LAYER}
batch_size: {BATCH_SIZE}, train_seqeunces: {train_seqeunces}, val_sequences: {val_sequences}, RealEstate: {USE_REALESTATE}, batchnorm & dropout: {BN_AND_DO}, 
average embeddings: {AVG_EMBEDDINGS}, model: {MODEL}, augmentation: {AUGMENTATION}, 
predict pose: {PREDICT_POSE}, SVD coeff: {LAST_SV_COEFF}, RE1 coeff: {re1_coeff} SED coeff: {sed_coeff}, ALG_COEFF: {alg_coeff}, unforzen layers: {UNFROZEN_LAYERS}, group conv: {GROUP_CONV}\n\n"""
        print_and_write(parameters, plots_path)

        model.train_model(train_loader, val_loader, num_epochs=NUM_EPOCHS)




