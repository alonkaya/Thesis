import torch

DEVICE_ID = 1
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

### Dataset ###
CROP = 224
RESIZE = 256
ANGLE_RANGE = 90
SHIFT_RANGE = 110
train_length = 9216   # Needs to be a multiple of batch size
val_length = 1600      # Needs to be a multiple of batch size
test_length = 1600     # Needs to be a multiple of batch size

### Training ###
LR = [6e-5]
BATCH_SIZE = [32]
NORM = True
TRAIN_FROM_SCRATCH = False
NUM_WORKERS = 0 # Probably setting this to > 0 causes Nans. If you get Nans then set it to 0.
SAVE_MODEL = True
NUM_EPOCHS = 300
ADDITIONS = ""                                      
GET_OLD_PATH = False
SEED = 42
ALPHA = [10]
EMBEDDINGS_TO_USE = [["rotated_embeddings", "original_embeddings"]]

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
NUM_OUTPUT = 3
FROZEN_LAYERS = 4
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
# only one of the following 3 should be True
USE_CONV = True
AVG_EMBEDDINGS = False
USE_CLS = [False]
