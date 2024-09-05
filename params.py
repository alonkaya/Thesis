import torch

DEVICE_ID = 1
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

### Dataset ###
CROP = 224
RESIZE = 256
angle_range = 90
shift_range = 150
AUGMENTATION = True
RANDOM_CROP = True

### Training ###
LR = [1e-4]                                                               # TODO lr: 5e-4, 1e-4, 5e-5, 2e-5
BATCH_SIZE = [4]                                                          # TODO 16, 32, 64
NORM = True
TRAIN_FROM_SCRATCH = False
NUM_WORKERS = 0 
SAVE_MODEL = True
NUM_EPOCHS = 1500
ADDITIONS = ""                                     
GET_OLD_PATH = False
SEED = 42
L2_COEFF = 1
HUBER_COEFF = 1                                                      # TODO: coeffs (1,1), (0.5,0.5), (0.1,0.1), (0.1,1), (1,0.1)

#### Model ###
MLP_HIDDEN_DIM = [1024, 512]
CONV_HIDDEN_DIM = [256, 512]
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
# CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
VIT_MODEL_NAME = "google/vit-base-patch32-224-in21k"
PRETRAINED_PATH =  None # make sure you set GET_OLD_PATH !! 
RESNET_MODEL_NAME = 'microsoft/resnet-152'
MODEL = CLIP_MODEL_NAME
FREEZE_PRETRAINED_MODEL=False
AVG_EMBEDDINGS = False
USE_CONV = True
NUM_OUTPUT = 3
FROZEN_LAYERS = [0]
norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device) if MODEL == CLIP_MODEL_NAME else torch.tensor([0.5, 0.5, 0.5]).to(device)
norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device) if MODEL == CLIP_MODEL_NAME else torch.tensor([0.5, 0.5, 0.5]).to(device)

