import torch

# DEVICE_ID = 1
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

### Dataset ###
norm_mean = torch.tensor([0.449, 0.449, 0.449])
norm_std = torch.tensor([0.226, 0.226, 0.226])
train_seqeunces = [0, 2, 3, 5, 6, 7, 8]
val_sequences = [0, 2, 3, 5, 6, 7, 8]
train_seqeunces_stereo = [0, 2, 3, 5]
val_sequences_stereo = [6, 7, 8]
CROP = 224
RESIZE = 256
USE_REALESTATE = False
STEREO = True
RIGHTCAMVAL = False
VAL_LENGTH = 900
FIRST_2_THRIDS_TRAIN = False
FIRST_2_OF_3_TRAIN = False
JUMP_FRAMES = 6 if USE_REALESTATE else 2
AUGMENTATION = True
RANDOM_CROP = True

### Training ###
learning_rates_vit = [2e-5]
learning_rates_mlp = [2e-5]
USE_RECONSTRUCTION_LAYER = True
BATCH_SIZE = 1 # TODO:  change pose_to_F if batch size > 1 ! 
TRAIN_FROM_SCRATCH = False
DEEPF_NOCORRS = False
IMAGE_TYPE = "jpg" if USE_REALESTATE else "png"
NUM_WORKERS = 0 # Change Main.py if > 0
BN_AND_DO = True if BATCH_SIZE > 1 else False
SAVE_MODEL = True
NUM_EPOCHS = 200
VISIUALIZE = {"epoch" : -1, "dir": 'predicted_epipole_lines'}
ADDITIONS = ""

### Epipolar geometry ###
RE1_DIST = True
SED_DIST = True
SED_BAD_THRESHOLD = 0.01 if STEREO else 0.1
EPIPOLAR_THRESHOLD = 0.3 if STEREO else 0.22
SED_TRIM_THRESHOLD = 0.01 if STEREO else 0.1
LAST_SV_COEFF = 0 if USE_RECONSTRUCTION_LAYER else 1
ALG_COEFF = [0.1]
RE1_COEFF = [0]
SED_COEFF = [0]

#### Model ###
MLP_HIDDEN_DIM = [1024, 512, 256]
CONV_HIDDEN_DIM = [1024, 2048, 1024, 512]
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
# CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
VIT_MODEL_NAME = "google/vit-base-patch32-224-in21k"
PRETRAINED_PATH = None
MODEL = CLIP_MODEL_NAME
FREEZE_PRETRAINED_MODEL=False
AVG_EMBEDDINGS = True
USE_CONV = False
GROUP_CONV = {"use" : False, "out_channels": 256}
NUM_OUTPUT = 8 if USE_RECONSTRUCTION_LAYER else 9
UNFROZEN_LAYERS = 0

