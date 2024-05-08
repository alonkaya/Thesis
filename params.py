import torch

DEVICE_ID = 0
# if DEVICE_ID==0: DEVICE_ID=1
# elif DEVICE_ID==2: DEVICE_ID=1
# elif DEVICE_ID==1: DEVICE_ID==0
device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")

learning_rates_vit = [2e-5]
learning_rates_mlp = [2e-5]
train_seqeunces = [0,1,2,3,4]
val_sequences = [0,1,2,3,4]
VAL_LENGTH = 600
norm_mean = torch.tensor([0.449, 0.449, 0.449])
norm_std = torch.tensor([0.226, 0.226, 0.226])

BATCH_SIZE = 1 # TODO:  change pose_to_F if batch size > 1 ! 
USE_REALESTATE = False
JUMP_FRAMES = 6 if USE_REALESTATE else 2
MLP_HIDDEN_DIM = [512, 256]
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
VIT_MODEL_NAME = "google/vit-base-patch32-224-in21k"
EPIPOLAR_THRESHOLD = 0.3
DEEPF_NOCORRS = False
IMAGE_TYPE = "jpg" if USE_REALESTATE else "png"
NUM_WORKERS = 0 # Change Main.py if > 0
BN_AND_DO = True if BATCH_SIZE > 1 else False
SAVE_MODEL = True
SED_BAD_THRESHOLD = 0.06

SED_TRIM_THRESHOLD = 0.1
RE1_DIST = True
SED_DIST = True
USE_RECONSTRUCTION_LAYER = True
LAST_SV_COEFF = 0 if USE_RECONSTRUCTION_LAYER else 1
ALG_COEFF = [0]
RE1_COEFF = [0]
SED_COEFF = [0.1]
PREDICT_POSE = False
NUM_OUTPUT = 8 if USE_RECONSTRUCTION_LAYER else 9
NUM_EPOCHS = 100
MODEL = CLIP_MODEL_NAME
AUGMENTATION = True
RANDOM_CROP = True
FREEZE_PRETRAINED_MODEL=False
AVG_EMBEDDINGS = True
UNFROZEN_LAYERS = 0
GROUP_CONV = {"use" : False, "out_channels": 256}
VISIUALIZE = {"epoch" : -1, "dir": 'predicted_epipole_lines'}
FIRST_2_THRIDS_TRAIN = False
FIRST_2_OF_3_TRAIN = False
ADDITIONS = "RightCamVal__"