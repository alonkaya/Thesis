import torch

DEVICE_ID = 1
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

### Dataset ###
CROP = 224
RESIZE = 256
ANGLE_RANGE = 90
SHIFT_RANGE = 0
train_length = 4096 # Needs to be a multiple of batch size
val_length = 640 # Needs to be a multiple of batch size
test_length = 640 # Needs to be a multiple of batch size

### Training ###
LR = [4e-5, 5e-4, 1e-4]                                                               # TODO lr: 5e-4, 1e-4, 5e-5, 2e-5
BATCH_SIZE = [8]                                                                     # TODO 16, 32, 64
NORM = True
TRAIN_FROM_SCRATCH = False
NUM_WORKERS = 2
SAVE_MODEL = True
NUM_EPOCHS = 50
ADDITIONS = ""                                     
GET_OLD_PATH = False
SEED = 42
ALPHA = [1]

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
USE_CONV = False
AVG_EMBEDDINGS = True
NUM_OUTPUT = 1
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

