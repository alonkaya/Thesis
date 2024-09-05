# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from Dataset import get_data_loaders
from utils import print_and_write, init_main
from FMatrixRegressor import FMatrixRegressor
from params import *
import itertools
import argparse

if __name__ == "__main__":
        init_main()

        # Iterate over each combination
        param_combinations = itertools.product(ALG_COEFF, RE1_COEFF, SED_COEFF, seq_ratios, lrs, batch_size, part, frozen_layers)

        for i, (alg_coeff, re1_coeff, sed_coeff, data_ratio, lr, bs, part, fl) in enumerate(param_combinations):
                set_seed(SEED)

                coeff = f'ALG_sqr_{alg_coeff}__' if alg_coeff > 0 else f'RE1_{re1_coeff}__' if re1_coeff > 0 else f'SED_{sed_coeff}__' if sed_coeff > 0 else ''
                dataset_class = "__first_2_thirds_train" if FIRST_2_THRIDS_TRAIN else "__first_2_of_three_train" if FIRST_2_OF_3_TRAIN else ""
                dataset = 'DeepF_noCors' if DEEPF_NOCORRS else 'Stereo' if STEREO else 'RealEstate' if USE_REALESTATE else 'KITTI_RightCamVal' if RIGHTCAMVAL else 'KITTI'
                scratch = 'Scratch__' if TRAIN_FROM_SCRATCH else ''
                enlarged_clip = 'Enlarged__' if MODEL == "openai/clip-vit-large-patch14" else ""
                model = "CLIP" if MODEL == CLIP_MODEL_NAME else "Resnet" if MODEL == RESNET_MODEL_NAME else "Google ViT" 
                compress = f'avg_embeddings' if AVG_EMBEDDINGS else f'conv'

                plots_path = os.path.join('plots', dataset, 'Winners',
                                        f"""{coeff}L2_{L2_coeff}__huber_{huber_coeff}__lr_{lr}__{compress}__{model}__use_reconstruction_{USE_RECONSTRUCTION_LAYER}""",  \
                                        f"""BS_{bs}__ratio_{data_ratio}__{part}__frozen_{fl}{ADDITIONS}""")
                
   
                train_loader, val_loader, test_loader = get_data_loaders(data_ratio, part, batch_size=bs)

                model = FMatrixRegressor(lr=lr, lr_decay=lr_decay, min_lr=MIN_LR, batch_size=bs, L2_coeff=L2_coeff, huber_coeff=huber_coeff, alg_coeff=alg_coeff, re1_coeff=re1_coeff, sed_coeff=sed_coeff, plots_path=plots_path, pretrained_path=PRETRAINED_PATH, num_epochs=num_epochs, frozen_layers=fl).to(device)

                if model.start_epoch < model.num_epochs:
                        parameters = f"""###########################################################################################################################################################\n
                        {ADDITIONS} learning rate: {lr}, lr_decay: {lr_decay}, mlp_hidden_sizes: {MLP_HIDDEN_DIM}, jump_frames: {JUMP_FRAMES}, use_reconstruction_layer: {USE_RECONSTRUCTION_LAYER}
                        batch_size: {bs}, norm: {NORM}, train_seqeunces: {train_seqeunces_stereo if STEREO else train_seqeunces}, val_sequences: {val_sequences_stereo if STEREO else val_sequences}, dataset: {dataset},
                        average embeddings: {AVG_EMBEDDINGS}, model: {MODEL}, augmentation: {AUGMENTATION}, random crop: {RANDOM_CROP}, deepF_nocorrs: {DEEPF_NOCORRS}, part: {part}, get_old_path: {GET_OLD_PATH},
                        SVD coeff: {LAST_SV_COEFF}, RE1 coeff: {re1_coeff} SED coeff: {sed_coeff}, ALG_COEFF: {alg_coeff}, L2_coeff: {L2_coeff}, huber_coeff: {huber_coeff}, frozen layers: {fl},
                        crop: {CROP} resize: {RESIZE}, use conv: {USE_CONV} pretrained: {PRETRAINED_PATH}, data_ratio: {data_ratio}, norm_mean: {norm_mean}, norm_std: {norm_std}, sched: {SCHED} seed: {SEED}, \n\n"""
                        print_and_write(parameters, model.plots_path)
                        
                        if PRETRAINED_PATH or os.path.exists(os.path.join(plots_path, 'model.pth')):
                                print_and_write(f"##### CONTINUE TRAINING #####\n\n", model.plots_path)
                
                        model.train_model(train_loader, val_loader, test_loader)
                else: 
                        print(f"Model {plots_path} already trained")

                torch.cuda.empty_cache()

