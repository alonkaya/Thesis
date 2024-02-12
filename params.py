import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_sizes = [1,32]
learning_rates_vit = [8e-5, 4e-5, 1e-5]
learning_rates_mlp = [2e-5, 5e-5, 1e-4, 5e-4]
penalty_coeffs = [1, 0.1, 5]
train_seqeunces = [0,2]
val_sequences = [1,3,4]
penaltize_normalized_options = [False]

USE_REALESTATE = True
JUMP_FRAMES = 8 if USE_REALESTATE else 2
NUM_EPOCHS = 20
MLP_HIDDEN_DIM = [512, 256]
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
SHOW_PLOTS = False
USE_RECONSTRUCTION_LAYER = False
NUM_OUTPUT = 8 if USE_RECONSTRUCTION_LAYER else 9
EPIPOLAR_THRESHOLD = 0.15
DEEPF_NOCORRS = False
MOVE_BAD_IMAGES = False
IMAGE_TYPE = "jpg"  if USE_REALESTATE else "png"
BATCHNORM_AND_DROPOUT_OPTIONS = [True, False]
NUM_WORKERS = 0