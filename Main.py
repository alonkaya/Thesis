# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from Dataset import get_data_loaders
from utils import print_and_write, init_main, set_seed
from FMatrixRegressor import FMatrixRegressor
from params import *
import itertools
import argparse
import sys
import shutil

if __name__ == "__main__":
        init_main()

        parser = argparse.ArgumentParser()

        parser.add_argument("--bs", type=int, default=BATCH_SIZE)
        parser.add_argument("--lr", nargs="+", type=float, default=LR)
        parser.add_argument("--l2", type=float, default=L2_COEFF)    
        parser.add_argument("--huber", type=float, default=HUBER_COEFF)
        parser.add_argument("--fl", nargs="+", type=int)
        parser.add_argument("--parts", nargs="+", type=str)
        parser.add_argument("--dr", nargs="+", type=float)

        args = parser.parse_args()

        batch_size = args.bs
        lrs = args.lr
        L2_coeff = args.l2
        huber_coeff = args.huber
        frozen_layers = args.fl if args.fl else FROZEN_LAYERS

        parts = PART if not SCENEFLOW else [None]
        train_sizes = SEQ_RATIOS if STEREO else RL_TRAIN_NUM
        
        gtg = False
        # Iterate over each combination
        param_combinations = itertools.product(ALG_COEFF, RE1_COEFF, SED_COEFF, SEED, train_sizes, lrs, parts, frozen_layers)
        for i, (alg_coeff, re1_coeff, sed_coeff, seed, train_size, lr, part, fl) in enumerate(param_combinations):
                set_seed(seed)
                if STEREO and not SCENEFLOW and part != "head" and part != "mid" and part != "tail":
                        raise ValueError("Invalid part")
                
                if SCENEFLOW and FLYING:
                        num_epochs = 8000 if train_size==170 else 18000 if 80 else 45000 if train_size==9 else 0
                else:
                        num_epochs = 2000 if train_size==0.3 else 4500 if train_size==0.2 else 7000 if train_size==0.1 else \
                                     14000 if train_size==0.05 else 18000 if train_size==0.0375 else 24000 if train_size==0.025 else \
                                     40000 if train_size==0.015 else 65000 if train_size==0.008 else 100000 if train_size==0.004 else 0
                if num_epochs == 0:
                        print("Invalid data ratio")
                        continue
                
                if (not PRETEXT_TRAIN and MODEL==CLIP_MODEL_NAME and not SCENEFLOW) and (train_size==0.05 or (train_size==0.025 and part=="head" and fl==0)):
                        batch_size = 4 
                else:
                        batch_size = args.bs

                frozen_high_layers = 0 if fl > 0 else FROZEN_HIGH_LAYERS

                coeff = f'ALG_sqr_{alg_coeff}__' if alg_coeff > 0 else f'RE1_{re1_coeff}__' if re1_coeff > 0 else f'SED_{sed_coeff}__' if sed_coeff > 0 else ''
                dataset = 'Kitti2Flying' if KITTI2SCENEFLOW and FLYING and SCENEFLOW else 'Flying' if FLYING and SCENEFLOW else 'Kitti2Monkaa' if KITTI2SCENEFLOW and SCENEFLOW else 'Monkaa' if SCENEFLOW else 'Stereo' if STEREO else 'RealEstate_split' if USE_REALESTATE and REALESTATE_SPLIT else 'RealEstate' if USE_REALESTATE else 'KITTI_RightCamVal' if RIGHTCAMVAL else 'KITTI'
                scratch = 'Scratch__' if TRAIN_FROM_SCRATCH else ''
                enlarged_clip = 'Enlarged__' if MODEL == "openai/clip-vit-large-patch14" else ""
                model = "CLIP" if MODEL == CLIP_MODEL_NAME else "Resnet" if MODEL == RESNET_MODEL_NAME else "Google ViT" 
                compress = f'avg_embeddings' if AVG_EMBEDDINGS else 'conv' if USE_CONV else 'all_embeddings'
                seed_param = "" if seed == 42 else f"__seed_{seed}"
                data_config = f'ratio_{train_size}__{part}' if not SCENEFLOW else f'ratio_{train_size}'
                frozen = fl if frozen_high_layers==0 else f'{frozen_high_layers}_high'
                plots_path = os.path.join('plots', dataset, 'Winners' if STEREO and not SCENEFLOW else '',
                                        f"""{coeff}L2_{L2_coeff}__huber_{huber_coeff}__lr_{lr}__{compress}__{model}__use_reconstruction_{USE_RECONSTRUCTION_LAYER}""",  \
                                        "Trained_vit" if TRAINED_VIT else "", \
                                        f"""BS_{batch_size}__{data_config}__frozen_{frozen}{ADDITIONS}{seed_param}""")
                
                if ONLY_CONTINUE:
                        model_path = os.path.join("/mnt/sda2/Alon", plots_path, "model.pth") if COMPUTER==0 else os.path.join(plots_path, "model.pth")
                        if not os.path.exists(model_path):
                                print(f'no model for {model_path}')
                                continue

                train_loader, val_loader, test_loader = get_data_loaders(train_size, part, batch_size=batch_size)

                model = FMatrixRegressor(lr=lr, min_lr=MIN_LR, batch_size=batch_size, L2_coeff=L2_coeff, huber_coeff=huber_coeff, alg_coeff=alg_coeff, re1_coeff=re1_coeff, sed_coeff=sed_coeff, \
                                         plots_path=plots_path, trained_vit=TRAINED_VIT, pretrained_path=PRETRAINED_PATH, num_epochs=num_epochs, frozen_layers=fl, frozen_high_layers=frozen_high_layers).to(device)

                # If the model was bad trained, skip it
                if os.path.exists((f'{model.plots_path}__bad')):
                        print(f"\n###\n{model.plots_path}\nAlready trained and got bad results\n###\n")   
                        sys.stdout.flush()
     
                # If the model was already trained WELL with seed 42, skip training with other seed
                elif "seed" in model.plots_path and os.path.exists(model.plots_path.split("__seed_")[0]):
                        print(f"\n###\n{model.plots_path}\nSeed 42 already well trained, no need for other seed training\n###\n")
                        sys.stdout.flush()

                elif model.start_epoch < model.num_epochs+10000:
                        parameters = f"""\n###########################################################################################################################################################\n
{ADDITIONS} learning rate: {lr}, mlp_hidden_sizes: {MLP_HIDDEN_DIM}, jump_frames: {JUMP_FRAMES}, use_reconstruction_layer: {USE_RECONSTRUCTION_LAYER}
batch_size: {batch_size}, norm: {NORM}, train_seqeunces: {train_seqeunces_stereo}, val_sequences: {val_sequences_stereo}, RL_TEST_NAMES: {RL_TEST_NAMES}, dataset: {dataset},
average embeddings: {AVG_EMBEDDINGS}, model: {MODEL}, augmentation: {AUGMENTATION}, random crop: {RANDOM_CROP}, part: {part}, get_old_path: {GET_OLD_PATH}, computer: {COMPUTER},
RE1 coeff: {re1_coeff} SED coeff: {sed_coeff}, ALG_COEFF: {alg_coeff}, L2_coeff: {L2_coeff}, huber_coeff: {huber_coeff}, frozen layers: {fl}, trained vit: {TRAINED_VIT},
crop: {CROP} resize: {RESIZE}, use conv: {USE_CONV} pretrained: {PRETRAINED_PATH}, train_size: {train_size}, norm_mean: {norm_mean}, norm_std: {norm_std}, sched: {SCHED} seed: {seed}\n"""
                        print_and_write(parameters, model.plots_path)
                        
                        print_and_write(f'train size: {len(train_loader.dataset)}, val size: {len(val_loader.dataset)}, test size: {len(test_loader.dataset)}\n', plots_path)

                        if PRETRAINED_PATH or os.path.exists(os.path.join(plots_path, 'model.pth')) or (os.path.exists(model.parent_model_path) and os.path.exists(os.path.join(model.parent_model_path, 'model.pth'))):
                                print_and_write(f"##### CONTINUE TRAINING #####\n", model.plots_path)
                
                        model.train_model(train_loader, val_loader, test_loader)

                        if COMPUTER==0:
                                os.makedirs(model.parent_model_path, exist_ok=True)
                                source_model_path = os.path.join(model.plots_path, 'model.pth')
                                dest_model_path = os.path.join(model.parent_model_path, 'model.pth')
                                try:
                                        shutil.move(source_model_path, dest_model_path)
                                except Exception as e:
                                        print(f"########\nError moving model to mnt from {source_model_path}\n{e}")

                        if os.path.exists(os.path.join(model.plots_path, 'backup_model.pth')):
                                os.remove(os.path.join(model.plots_path, 'backup_model.pth'))
                        else:
                                print_and_write(f"\n{model.plots_path} no backup to delete after job done\n", model.plots_path)

                else: 
                        print(f"Model {plots_path} already trained")
                        sys.stdout.flush()
                
                del train_loader, val_loader, test_loader, model
                torch.cuda.empty_cache()

