import torch
device, RESNET_MODEL_NAME, CLIP_MODEL_NAME = torch.device(f"cuda" if torch.cuda.is_available() else "cpu"), 'microsoft/resnet-152', "openai/clip-vit-base-patch32"

# 1242969 output_0.0375_0.025_frozen_8.log
# 1483631 output_0.015_orig_25000.log
# 1580251 output_0.015_pretrained_25000_frozen_0_4.log
# 1675862 output_resnet_0.015_25000.log
# 1297205 output_4090_best_100_orig.log in 4090

# nohup env CUDA_VISIBLE_DEVICES=0 TORCH_USE_CUDA_DSA=1 python Main.py > output_.log 2>&1 &
# gpuQ.py submit -d any -p /home/alonkay/Thesis -e alon_env -c "python Main.py  > output_.log 2>&1"

USE_REALESTATE = False
STEREO = True
PRETEXT_TRAIN = True
MODEL = CLIP_MODEL_NAME 
FROZEN_LAYERS = [0] if MODEL==RESNET_MODEL_NAME or USE_REALESTATE else [0,4,8]

### Dataset ###  
RIGHTCAMVAL = False
CROP = 224
RESIZE = 256
AUGMENTATION = True
RANDOM_CROP = True
INIT_DATA = True 

### STEREO ###
train_seqeunces_stereo = [0,2,3,5] #  10840 images 
val_sequences_stereo =  [6,7,8]    #  3682 images
test_sequences_stereo = [9]        #  1064 images
SEQ_RATIOS = [0.2, 0.1, 0.05, 0.0375, 0.025, 0.015]     # 3251, 2166, 1082, 540, 405, 269, 161                                                    
PART = ["head", "mid", "tail"] 

### RealEstate ###
RL_TEST_NAMES = ["fe2fadf89a84e92a", "f01e8b6f8e10fdd9", "f1ee9dc6135e5307", "a41df4fa06fd391b", "bc0ebb7482f14795", "9bdd34e784c04e3a", "98ebee1c36ecec55"]  # val 656, test 704
JUMP_FRAMES = 6 
RL_TRAIN_SPLIT_RATIO = 0.7
RL_TRAIN_NUM = [50]   #  14=1872  #  18=2136  #  20=2368  #  50=6560
REALESTATE_SPLIT = False # 50=4632

### Training ###
MIN_LR = 2e-5
SCHED = None
USE_RECONSTRUCTION_LAYER = True
NORM = True
TRAIN_FROM_SCRATCH = False
IMAGE_TYPE = "jpg" if USE_REALESTATE else "png"
NUM_WORKERS = 0 
SAVE_MODEL = True
GET_OLD_PATH = False
SEED = [42, 300, 500]
LR = [1e-4]             
BATCH_SIZE = [8]   

### Epipolar geometry ###
RE1_DIST = True
SED_DIST = True
EPIPOLAR_THRESHOLD = 0.3 
SED_TRIM_THRESHOLD = 0.01 if STEREO else 0.02
ALG_COEFF = [0]
RE1_COEFF = [0]
SED_COEFF = [0.5]                                                    
L2_COEFF = 1
HUBER_COEFF = 1                                                      
ADDITIONS = ""                                     

#### Model ###
MLP_HIDDEN_DIM = [1024, 512]
CONV_HIDDEN_DIM = [256, 512]
VIT_MODEL_NAME = "google/vit-base-patch32-224-in21k"
PRETRAINED_PATH =  None # make sure you set GET_OLD_PATH !! 
TRAINED_VIT = None if MODEL==RESNET_MODEL_NAME or USE_REALESTATE or not PRETEXT_TRAIN else "plots/Affine/BS_32__lr_6e-05__train_size_9216__CLIP__alpha_10__conv__original_rotated/model.pth" # This is for when wanting to fine-tune an already trained vit (for example fine-tuning a vit which had been trained on the affine transfomration task)
AVG_EMBEDDINGS = False
USE_CONV = True
NUM_OUTPUT = 8 if USE_RECONSTRUCTION_LAYER else 9
norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device) if MODEL == CLIP_MODEL_NAME else torch.tensor([0.5, 0.5, 0.5]).to(device)
norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device) if MODEL == CLIP_MODEL_NAME else torch.tensor([0.5, 0.5, 0.5]).to(device)
