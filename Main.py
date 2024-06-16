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
from DatasetOneSequence import * 


if __name__ == "__main__":
    init_main()

    # Iterate over each combination
    param_combinations = itertools.product(learning_rates_vit, learning_rates_mlp, ALG_COEFF, RE1_COEFF, SED_COEFF)

    for i, (lr_vit, lr_mlp, alg_coeff, re1_coeff, sed_coeff) in enumerate(param_combinations):

        coeff = f'ALG_sqr_{alg_coeff}__' if alg_coeff > 0 else f'RE1_{re1_coeff}__' if re1_coeff > 0 else f'SED_{sed_coeff}__' if sed_coeff > 0 else ''
        dataset_class = "__first_2_thirds_train" if FIRST_2_THRIDS_TRAIN else "__first_2_of_three_train" if FIRST_2_OF_3_TRAIN else ""
        dataset = 'DeepF_noCors' if DEEPF_NOCORRS else 'Stereo' if STEREO else 'RealEstate' if USE_REALESTATE else 'KITTI_RightCamVal' if RIGHTCAMVAL else 'KITTI'
        scratch = 'Scratch__' if TRAIN_FROM_SCRATCH else ''
        continued = "Continued__" if PRETRAINED_PATH else ""
        enlarged_clip = 'Enlarged__' if MODEL == "openai/clip-vit-large-patch14" else ""
        
        plots_path = os.path.join('plots', dataset, 
                          f"""{coeff}{ADDITIONS}{scratch}{enlarged_clip}{continued}lr_{learning_rates_mlp[0]}__\
avg_embeddings_{AVG_EMBEDDINGS}__conv_{USE_CONV}__model_{"CLIP" if MODEL == CLIP_MODEL_NAME else "Google ViT"}__\
use_reconstruction_{USE_RECONSTRUCTION_LAYER}__Augment_{AUGMENTATION}__rc_{RANDOM_CROP}__BS_{BATCH_SIZE}{'__small' if SMALL_DATA else ''}{dataset_class}""")\
        
        model = FMatrixRegressor(lr_vit=lr_vit, lr_mlp=lr_mlp, alg_coeff=alg_coeff, re1_coeff=re1_coeff, sed_coeff=sed_coeff, plots_path=plots_path, pretrained_path=PRETRAINED_PATH).to(device)

        train_loader, val_loader, test_loader = get_data_loaders(BATCH_SIZE)
        
        if not PRETRAINED_PATH:
                parameters = f"""###########################################################################################################################################################\n
        {ADDITIONS}learning rate vit: {lr_vit}, learning rate mlp: {lr_mlp}, lr_sched: {SCHED}, mlp_hidden_sizes: {MLP_HIDDEN_DIM}, jump_frames: {JUMP_FRAMES}, use_reconstruction_layer: {USE_RECONSTRUCTION_LAYER}
        batch_size: {BATCH_SIZE}, norm: {NORM}, train_seqeunces: {train_seqeunces_stereo if STEREO else train_seqeunces}, val_sequences: {val_sequences_stereo if STEREO else val_sequences}, dataset: {dataset},
        average embeddings: {AVG_EMBEDDINGS}, model: {MODEL}, augmentation: {AUGMENTATION}, random crop: {RANDOM_CROP}, deepF_nocorrs: {DEEPF_NOCORRS},
        SVD coeff: {LAST_SV_COEFF}, RE1 coeff: {re1_coeff} SED coeff: {sed_coeff}, ALG_COEFF: {alg_coeff}, unforzen layers: {UNFROZEN_LAYERS}, group conv: {GROUP_CONV["use"]}
        crop: {CROP} resize: {RESIZE}, use conv: {USE_CONV} pretrained: {PRETRAINED_PATH} {continued}\n\n"""
                print_and_write(parameters, model.plots_path)

        model.train_model(train_loader, val_loader, test_loader)




