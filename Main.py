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
        set_seed(SEED)
                
        # Iterate over each combination
        param_combinations = itertools.product(LR, BATCH_SIZE, ALPHA, EMBEDDINGS_TO_USE, USE_CLS, train_length)

        for i, (lr, bs, alpha, embeddings_to_use, use_cls, size) in enumerate(param_combinations):
                scratch = 'Scratch__' if TRAIN_FROM_SCRATCH else ''
                enlarged_clip = 'Enlarged__' if MODEL == "openai/clip-vit-large-patch14" else ""
                model = "CLIP" if MODEL == CLIP_MODEL_NAME else "CLIP_16" if MODEL==CLIP_MODEL_NAME_16 else "Resnet" if MODEL == RESNET_MODEL_NAME else "Google ViT" 
                regress = 'avg' if AVG_EMBEDDINGS else 'conv' if USE_CONV else 'cls' if use_cls else 'all_patches'
                which = "angle" if  SHIFT_RANGE==0 else "shift" if ANGLE_RANGE==0 else "angle_shift"
                embeddings = 'all' if len(embeddings_to_use)==3 else 'original_rotated' if len(embeddings_to_use)==2 \
                                    else 'rotated' if embeddings_to_use[0] == "rotated_embeddings" else 'mul'
                frozen = 'all' if FREEZE_PRETRAINED_MODEL else FROZEN_LAYERS
                plots_path = os.path.join('plots', 'Affine', f'BS_{bs}__lr_{lr}__alpha_{alpha}__{regress}__{embeddings}_angle_{ANGLE_RANGE}__shift_{SHIFT_RANGE}', \
                                          model, f'size_{size}__frozen_{frozen}{ADDITIONS}')
   
                train_loader, val_loader, test_loader = get_dataloaders(batch_size=bs, train_length=size, val_length=val_length, test_length=test_length, plots_path=plots_path)
                

                bs = 4 if size < 1000 else bs
                num_epochs = 400 if size==4048 else 800 if size==1048 else 1000 if size==256 else 1400 if size==64 else 0
                model = AffineRegressor(lr, bs, alpha, embeddings_to_use, use_cls, model_name=MODEL, avg_embeddings=AVG_EMBEDDINGS, plots_path=plots_path, \
                                        frozen_layers=FROZEN_LAYERS, pretrained_path=PRETRAINED_PATH, use_conv=USE_CONV, num_epochs=num_epochs)

                if model.start_epoch < model.num_epochs:
                        parameters = f"""\n###########################################################################################################################################################\n
                        {ADDITIONS} learning rate: {lr},  mlp_hidden_sizes: {MLP_HIDDEN_DIM}, batch_size: {bs}, norm: {NORM}, alpha: {alpha}, avg embeddings: {AVG_EMBEDDINGS}, 
                        crop: {CROP} resize: {RESIZE}, use conv: {USE_CONV} pretrained: {PRETRAINED_PATH}, seed: {SEED}, angle range: {ANGLE_RANGE}, shift range: {SHIFT_RANGE}, 
                        train length: {size}, val length: {val_length}, test length: {test_length}, get old path: {GET_OLD_PATH}, embeddings to use: {embeddings_to_use},
                        use_cls: {use_cls}, frozen layer: {FROZEN_LAYERS}\n\n"""
                        print_and_write(parameters, model.plots_path)
                        
                        if PRETRAINED_PATH or os.path.exists(os.path.join(plots_path, 'model.pth')):
                                print_and_write(f"##### CONTINUE TRAINING #####\n\n", model.plots_path)
                
                        model.train_model(train_loader, val_loader, test_loader)
                else: 
                        print(f"Model {plots_path} already trained")

                torch.cuda.empty_cache()
