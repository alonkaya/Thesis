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
USE_RECONSTRUCTION_LAYER = True
LAST_SV_COEFF = 0 if USE_RECONSTRUCTION_LAYER else 1
ALG_COEFF = [0]
RE1_COEFF = [0]
SED_COEFF = [0.1, 0.01, 0.001]
ENFORCE_RANK_2 = False
PREDICT_POSE = False
NUM_OUTPUT = 8 if USE_RECONSTRUCTION_LAYER else 9
NUM_EPOCHS = 100
MODEL = CLIP_MODEL_NAME
AUGMENTATION = False
FREEZE_PRETRAINED_MODEL=False
AVG_EMBEDDINGS = True
UNFROZEN_LAYERS = 0
GROUP_CONV = {"use" : False, "out_channels": 256, "num_groups" : 256}
VISIUALIZE = {"epoch" : -1, "dir": 'predicted_epipole_lines'}

