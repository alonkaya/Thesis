import torch

DEVICE_ID = 2
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

### Dataset ###
norm_mean = torch.tensor([0.449, 0.449, 0.449])
norm_std = torch.tensor([0.226, 0.226, 0.226])
train_seqeunces = [0, 2, 3, 5, 6, 7, 8]
val_sequences = [0, 2, 3, 5, 6, 7, 8]
FIRST_2_THRIDS_TRAIN = True
FIRST_2_OF_3_TRAIN = False
SMALL_DATA = True
train_seqeunces_stereo = [0,2,3,5] if not SMALL_DATA else [0]
val_sequences_stereo = train_seqeunces_stereo if FIRST_2_THRIDS_TRAIN else [6,7,8] if not SMALL_DATA else [6]
test_sequences_stereo = train_seqeunces_stereo if FIRST_2_THRIDS_TRAIN else [9]
CROP = 224
RESIZE = 256
USE_REALESTATE = False
STEREO = True
RIGHTCAMVAL = False
VAL_LENGTH = 600

JUMP_FRAMES = 6 if USE_REALESTATE else 2
AUGMENTATION = True
RANDOM_CROP = True

### Training ###
learning_rates_vit = [2e-5]
learning_rates_mlp = [2e-5]
USE_RECONSTRUCTION_LAYER = True
BATCH_SIZE = 16
NORM = False
TRAIN_FROM_SCRATCH = False
DEEPF_NOCORRS = False
IMAGE_TYPE = "jpg" if USE_REALESTATE else "png"
NUM_WORKERS = 0 # Change Main.py if > 0
SAVE_MODEL = True
NUM_EPOCHS = 1500
VISIUALIZE = {"epoch" : -1, "dir": 'predicted_epipole_lines'}
ADDITIONS = ""

### Epipolar geometry ###
RE1_DIST = True
SED_DIST = True
SED_BAD_THRESHOLD = 0.01 if STEREO else 0.1
EPIPOLAR_THRESHOLD = 0.3 if STEREO else 0.22
SED_TRIM_THRESHOLD = 0.01 if STEREO else 0.1
LAST_SV_COEFF = 0 if USE_RECONSTRUCTION_LAYER else 1
ALG_COEFF = [0]
RE1_COEFF = [0]
SED_COEFF = [0.05]

#### Model ###
MLP_HIDDEN_DIM = [1024, 512, 256]
CONV_HIDDEN_DIM = [1024, 512]
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
# CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
VIT_MODEL_NAME = "google/vit-base-patch32-224-in21k"
PRETRAINED_PATH = "plots/Stereo/SED_0.05__lr_2e-05__avg_embeddings_False__conv_True__model_CLIP__use_reconstruction_True__Augment_True__rc_True__BS_16__small__first_2_thirds_train"
MODEL = CLIP_MODEL_NAME
FREEZE_PRETRAINED_MODEL=False
AVG_EMBEDDINGS = False
USE_CONV = True
USE_CONV = False if DEEPF_NOCORRS else USE_CONV
GROUP_CONV = {"use" : False, "out_channels": 256}
NUM_OUTPUT = 8 if USE_RECONSTRUCTION_LAYER else 9
UNFROZEN_LAYERS = 0

