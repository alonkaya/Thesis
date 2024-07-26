import torch

DEVICE_ID = 2
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

### Dataset ###
train_seqeunces = [0, 2, 3, 5, 6, 7, 8]
val_sequences = [0, 2, 3, 5, 6, 7, 8]
FIRST_2_THRIDS_TRAIN = False
FIRST_2_OF_3_TRAIN = False
train_seqeunces_stereo = [0,2,3,5] #  9215 images 
val_sequences_stereo =  [6,7,8]    #  3697 images
test_sequences_stereo = [9]        #  1064 images
seq_ratios = [0.2, 0.1]      # 2764, 1843, 921, 460
CROP = 224
RESIZE = 256
USE_REALESTATE = False
STEREO = True
RIGHTCAMVAL = False
JUMP_FRAMES = 6 if USE_REALESTATE else 2
AUGMENTATION = True
RANDOM_CROP = True
INIT_DATA = True 
HEAD = True

### Training ###
LR = [5e-4, 1e-4]                                                             # TODO lr: 5e-4, 1e-4, 5e-5, 2e-5
WEIGHT_DECAY = 0                                                              # TODO 5e-4, 5e-5
MIN_LR = 2e-5
SCHED = None
USE_RECONSTRUCTION_LAYER = True
BATCH_SIZE = [16, 32]                                                         # TODO 16, 32, 64
NORM = True
TRAIN_FROM_SCRATCH = False
DEEPF_NOCORRS = False
IMAGE_TYPE = "jpg" if USE_REALESTATE else "png"
NUM_WORKERS = 0 
SAVE_MODEL = True
NUM_EPOCHS = 1500
VISIUALIZE = {"epoch" : -1, "dir": 'predicted_epipole_lines'}
ADDITIONS = "auged__"                                     

### Epipolar geometry ###
RE1_DIST = True
SED_DIST = True
SED_BAD_THRESHOLD = 0.01 if STEREO else 0.1
EPIPOLAR_THRESHOLD = 0.3 if STEREO else 0.22
SED_TRIM_THRESHOLD = 0.01 if STEREO else 0.1
LAST_SV_COEFF = 0 if USE_RECONSTRUCTION_LAYER else 1
ALG_COEFF = [0]
RE1_COEFF = [0]
SED_COEFF = [0.1, 0.5]                                                    # TODO 0.01, 0.05, 0.1, 0.5, 1
L2_COEFF = 1
HUBER_COEFF = 1                                                      # TODO: coeffs (1,1), (0.5,0.5), (0.1,0.1), (0.1,1), (1,0.1)

#### Model ###
MLP_HIDDEN_DIM = [1024, 512]
CONV_HIDDEN_DIM = [256, 512]
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
# CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
VIT_MODEL_NAME = "google/vit-base-patch32-224-in21k"
PRETRAINED_PATH = None
RESNET_MODEL_NAME = 'microsoft/resnet-152'
MODEL = CLIP_MODEL_NAME
FREEZE_PRETRAINED_MODEL=False
AVG_EMBEDDINGS = False
USE_CONV = True
USE_CONV = False if DEEPF_NOCORRS else USE_CONV
GROUP_CONV = {"use" : False, "out_channels": 256}
NUM_OUTPUT = 8 if USE_RECONSTRUCTION_LAYER else 9
UNFROZEN_LAYERS = 0
norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device) if MODEL == CLIP_MODEL_NAME else torch.tensor([0.5, 0.5, 0.5]).to(device)
norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device) if MODEL == CLIP_MODEL_NAME else torch.tensor([0.5, 0.5, 0.5]).to(device)


