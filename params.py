import torch
import os
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

learning_rates_vit = [2e-5]
learning_rates_mlp = [2e-5]
train_seqeunces = [0,2]
val_sequences = [1,3,4]
norm_mean = torch.tensor([0.449, 0.449, 0.449]).to(device)
norm_std = torch.tensor([0.226, 0.226, 0.226]).to(device)

BATCH_SIZE = 1 # TODO:  change pose_to_F if batch size > 1 ! 
USE_REALESTATE = True
JUMP_FRAMES = 6 if USE_REALESTATE else 2
MLP_HIDDEN_DIM = [512, 256]
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
VIT_MODEL_NAME = "google/vit-base-patch32-224-in21k"
EPIPOLAR_THRESHOLD = 0.18
DEEPF_NOCORRS = False
MOVE_BAD_IMAGES = False
IMAGE_TYPE = "jpg" if USE_REALESTATE else "png"
NUM_WORKERS = 0 # Change Main.py if > 0
BN_AND_DO = True if BATCH_SIZE > 1 else False
CUSTOMDATASET_TYPE = "CustomDataset_first_two_thirds_train"
SAVE_MODEL = True

RE1_DIST = True
SED_DIST = True
LAST_SV_COEFF = 1
ALG_COEFF = [0.1, 0.01, 0.001]
RE1_COEFF = 0
SED_COEFF = 0
ENFORCE_RANK_2 = False
USE_RECONSTRUCTION_LAYER = False
PREDICT_POSE = False
NUM_OUTPUT = 8 if USE_RECONSTRUCTION_LAYER else 9
NUM_EPOCHS = 1
MODEL = CLIP_MODEL_NAME
AUGMENTATION = False
FREEZE_PRETRAINED_MODEL=False
OVERFITTING=True
AVG_EMBEDDINGS = True
UNFROZEN_LAYERS = 0
GROUP_CONV = {"use" : False, "out_channels": 256, "num_groups" : 256}
VISIUALIZE = {"epoch" : NUM_EPOCHS-1, "dir": 'predicted_epipole_lines'}
PLOTS_PATH = os.path.join('plots', 'only_one_sequence', 
                          f"""SVD_{LAST_SV_COEFF}__RE1_{RE1_COEFF}__SED_{SED_COEFF}__ALG_{ALG_COEFF}__lr_{learning_rates_mlp[0]}__\
avg_embeddings_{AVG_EMBEDDINGS}__model_{"CLIP" if MODEL == CLIP_MODEL_NAME else "Google ViT"}__\
Force_rank_2_{ENFORCE_RANK_2}__predict_pose_{PREDICT_POSE}__use_reconstruction_{USE_RECONSTRUCTION_LAYER}""")
