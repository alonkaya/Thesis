import torch
device, RESNET_MODEL_NAME, CLIP_MODEL_NAME, CLIP_MODEL_NAME_16 = torch.device(f"cuda" if torch.cuda.is_available() else "cpu"), 'microsoft/resnet-152', "openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16"
# nohup env CUDA_VISIBLE_DEVICES=0 TORCH_USE_CUDA_DSA=1 python Main.py > output_.log 2>&1 &   ### REMEMBER TO FIRST MOVE THE MODEL FROM ORIGINAL PATH TO MNT PATH IN CASE OF COMPUTER==0 AND THE LAST RUN EXITED!!
# gpuQ.py submit -d any -p /home/alonkay/Thesis -e alon_env -c "python Main.py  > output_.log 2>&1"

MODEL = RESNET_MODEL_NAME

### Dataset ###
CROP = 224
RESIZE = 256
ANGLE_RANGE = 30
SHIFT_RANGE = 32
train_length = [4048, 1048, 256, 64]  # Needs to be a multiple of batch size
val_length = 320      # Needs to be a multiple of batch size
test_length = 320     # Needs to be a multiple of batch size
INIT_DATA = True
COMPUTER = 1 # 0=250, 1=146, 2=else

### Training ###
LR = [6e-5, 1e-4]
BATCH_SIZE = [32]
NORM = True
TRAIN_FROM_SCRATCH = False
NUM_WORKERS = 0 # Probably setting this to > 0 causes Nans. If you get Nans then set it to 0.
SAVE_MODEL = True
NUM_EPOCHS = 400
ADDITIONS = ""                                      
GET_OLD_PATH = False
SEED = 42
ALPHA = [10]
EMBEDDINGS_TO_USE = [["rotated_embeddings", "original_embeddings"]]
MAX_POOL_SIZE = 4 if not MODEL==CLIP_MODEL_NAME_16 else 7 

#### Model ###
MLP_HIDDEN_DIM = [1024, 512]
CONV_HIDDEN_DIM = [256, 512]
PRETRAINED_PATH =  None # make sure you set GET_OLD_PATH !! 
FREEZE_PRETRAINED_MODEL=False
NUM_OUTPUT = 3
FROZEN_LAYERS = 0
norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device) if MODEL == CLIP_MODEL_NAME or MODEL==CLIP_MODEL_NAME_16 else torch.tensor([0.5, 0.5, 0.5]).to(device)
norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device) if MODEL == CLIP_MODEL_NAME or MODEL==CLIP_MODEL_NAME_16 else torch.tensor([0.5, 0.5, 0.5]).to(device)
# only one of the following 3 should be True
USE_CONV = True
AVG_EMBEDDINGS = False
USE_CLS = [False]
