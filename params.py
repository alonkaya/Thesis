import torch

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
# 1179397 is output_resnet_0.025_0.0375_0.05.log
# 1179721 is output_vit_trained_0.1_0.2.log

### Dataset ###
train_seqeunces = [0, 2, 3, 5, 6, 7, 8]
val_sequences = [0, 2, 3, 5, 6, 7, 8]
train_seqeunces_stereo = [0,2,3,5] #  10840 images 
val_sequences_stereo =  [6,7,8]    #  3682 images
test_sequences_stereo = [9]        #  1064 images
SEQ_RATIOS = [0.2]      # 3251, 2166, 1082, 540, 405, 269
CROP = 224
RESIZE = 256
USE_REALESTATE = True
FM_NET = False
STEREO = False
RIGHTCAMVAL = False
JUMP_FRAMES = 6 if USE_REALESTATE else 2
AUGMENTATION = True
RANDOM_CROP = True
INIT_DATA = True
PART = ["head"]       
RealEstate_test_names = ["fe2fadf89a84e92a", "f01e8b6f8e10fdd9", "f1ee9dc6135e5307", "a41df4fa06fd391b", "bc0ebb7482f14795", "98ebee1c36ecec55"]                                           

### Training ###
LR = [1e-4]                                                               # TODO lr: 5e-4, 1e-4, 5e-5, 2e-5
WEIGHT_DECAY = 0                                                          # TODO 5e-4, 5e-5
MIN_LR = 2e-5
SCHED = None
USE_RECONSTRUCTION_LAYER = True
BATCH_SIZE = [8]                                                          # TODO 16, 32, 64
NORM = True
TRAIN_FROM_SCRATCH = False
IMAGE_TYPE = "jpg" if USE_REALESTATE or FM_NET else "png"
NUM_WORKERS = 0 
SAVE_MODEL = True
NUM_EPOCHS = 1500
GET_OLD_PATH = False
SEED = [42]

### Epipolar geometry ###
RE1_DIST = True
SED_DIST = True
SED_BAD_THRESHOLD = 0.01 if STEREO else 0.1
EPIPOLAR_THRESHOLD = 0.3 
SED_TRIM_THRESHOLD = 0.01 if STEREO else 0.02
LAST_SV_COEFF = 0 if USE_RECONSTRUCTION_LAYER else 1
ALG_COEFF = [0]
RE1_COEFF = [0]
SED_COEFF = [0.5]                                                    # TODO 0.01, 0.05, 0.1, 0.5, 1
L2_COEFF = 1
HUBER_COEFF = 1                                                      # TODO: coeffs (1,1), (0.5,0.5), (0.1,0.1), (0.1,1), (1,0.1)
ADDITIONS = ""                                     

#### Model ###
MLP_HIDDEN_DIM = [1024, 512]
CONV_HIDDEN_DIM = [256, 512]
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
# CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
VIT_MODEL_NAME = "google/vit-base-patch32-224-in21k"
PRETRAINED_PATH =  None # make sure you set GET_OLD_PATH !! 
TRAINED_VIT =  None # "plots/Affine/BS_32__lr_6e-05__train_size_9216__CLIP__alpha_10__conv__original_rotated/model.pth" # This is for when wanting to fine-tune an already trained vit (for example fine-tuning a vit which had been trained on the affine transfomration task)
RESNET_MODEL_NAME = 'microsoft/resnet-152'
MODEL = CLIP_MODEL_NAME
FREEZE_PRETRAINED_MODEL=False
AVG_EMBEDDINGS = False
USE_CONV = True
USE_CONV = False
NUM_OUTPUT = 8 if USE_RECONSTRUCTION_LAYER else 9
FROZEN_LAYERS = [0]
norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device) if MODEL == CLIP_MODEL_NAME else torch.tensor([0.5, 0.5, 0.5]).to(device)
norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device) if MODEL == CLIP_MODEL_NAME else torch.tensor([0.5, 0.5, 0.5]).to(device)
