# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from Dataset import get_dataloaders
from utils import print_and_write, init_main, set_seed
from AffineRegressor import AffineRegressor
from params import *
import itertools

if __name__ == "__main__":
        init_main()

        # Iterate over each combination
        param_combinations = itertools.product(LR, BATCH_SIZE, ALPHA)

        for i, (lr, bs, alpha) in enumerate(param_combinations):
                set_seed(SEED)
                
                scratch = 'Scratch__' if TRAIN_FROM_SCRATCH else ''
                enlarged_clip = 'Enlarged__' if MODEL == "openai/clip-vit-large-patch14" else ""
                model = "CLIP" if MODEL == CLIP_MODEL_NAME else "Resnet" if MODEL == RESNET_MODEL_NAME else "Google ViT" 

                plots_path = os.path.join('plots', f'BS_{bs}__lr_{lr}__train_size_{train_length}__model_{model}')
   
                train_loader, val_loader, test_loader = get_dataloaders(batch_size=bs, train_length=train_length, val_length=val_length, test_length=test_length)

                model = AffineRegressor(lr, bs, alpha, model_name=MODEL, plots_path=plots_path, pretrained_path=PRETRAINED_PATH, use_conv=USE_CONV, num_epochs=NUM_EPOCHS)

                if model.start_epoch < model.num_epochs:
                        parameters = f"""###########################################################################################################################################################\n
                        {ADDITIONS} learning rate: {lr},  mlp_hidden_sizes: {MLP_HIDDEN_DIM}, batch_size: {bs}, norm: {NORM},
                        crop: {CROP} resize: {RESIZE}, use conv: {USE_CONV} pretrained: {PRETRAINED_PATH}, seed: {SEED}, angle range: {ANGLE_RANGE}, shift range: {SHIFT_RANGE}, 
                        train length: {train_length}, val length: {val_length}, test length: {test_length}, get old path: {GET_OLD_PATH}\n\n\n"""
                        print_and_write(parameters, model.plots_path)
                        
                        if PRETRAINED_PATH or os.path.exists(os.path.join(plots_path, 'model.pth')):
                                print_and_write(f"##### CONTINUE TRAINING #####\n\n", model.plots_path)
                
                        model.train_model(train_loader, val_loader, test_loader)
                else: 
                        print(f"Model {plots_path} already trained")

                torch.cuda.empty_cache()

