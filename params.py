import torch
device, RESNET_MODEL_NAME, CLIP_MODEL_NAME = torch.device(f"cuda" if torch.cuda.is_available() else "cpu"), 'microsoft/resnet-152', "openai/clip-vit-base-patch32"
USE_REALESTATE = False
STEREO = True
# nohup env CUDA_VISIBLE_DEVICES=0 TORCH_USE_CUDA_DSA=1 python Main.py > output_.log 2>&1 &   ### REMEMBER TO FIRST MOVE THE MODEL FROM ORIGINAL PATH TO MNT PATH IN CASE OF COMPUTER==0 AND THE LAST RUN EXITED!!
# gpuQ.py submit -d any -p /home/alonkay/Thesis -e alon_env -c "python Main.py  > output_.log 2>&1"

# 4168282 output_clip_orig_frozen_8_tail.log
# 4168172 output_clip_orig_frozen_4_tail.log
# 4167851 output_clip_orig_frozen_0_tail.log

PRETEXT_TRAIN = False
SCENEFLOW = False
FLYING = False
MODEL = RESNET_MODEL_NAME 
FROZEN_LAYERS = [0] if MODEL==RESNET_MODEL_NAME or USE_REALESTATE else [6] if FLYING else [0,4,8]
COMPUTER = 0 # 0=132.72.49.250 1=else  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SEQ_RATIOS = [0.004,0.008,0.1,0.2] if not SCENEFLOW else [9] if FLYING else [1]     # 2166, 1082, 540, 405, 269, 161, 88, 47                                                 
KITTI2SCENEFLOW = False
ONLY_CONTINUE = True
PART = ["mid", "tail"] 
ADDITIONS = ""  ## REMEMBER TO PUT "__" !!!!!

### Dataset ###  
RIGHTCAMVAL = False
CROP = 224
RESIZE = 256
AUGMENTATION = True
RANDOM_CROP = True
INIT_DATA = True 
BATCH_SIZE = 8  

### STEREO KITTI ###
train_seqeunces_stereo = [0,2,3,5] #  10840 images 
val_sequences_stereo =  [6,7,8]    #  3682 images
test_sequences_stereo = [9]        #  1064 images

### SCENEFLOW MONKAA ###
train_seqeunces_monkaa =  ["treeflight_augmented0_x2", "treeflight_augmented1_x2", "lonetree_winter_x2", "a_rain_of_stones_x2", "eating_naked_camera2_x2",  "family_x2", "lonetree_difftex_x2"]  # 1035
val_sequences_monkaa = ["treeflight_x2", "eating_x2", "top_view_x2"] # 293
test_sequences_monkaa = ["flower_storm_x2", "funnyworld_x2", "eating_camera2_x2"]     # 375 frames   

### SCENEFLOW FLYING ###
test_sequences_flying = 100 

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

### Epipolar geometry ###
EPIPOLAR_THRESHOLD = 0.3 
SED_TRIM_THRESHOLD = 0.01 if STEREO or SCENEFLOW else 0.02
ALG_COEFF = [0]
RE1_COEFF = [0]
SED_COEFF = [0.5]                                                    
L2_COEFF = 1
HUBER_COEFF = 1                                                      

#### Model ###
MLP_HIDDEN_DIM = [1024, 512]
CONV_HIDDEN_DIM = [256, 512]
VIT_MODEL_NAME = "google/vit-base-patch32-224-in21k"
KITTI_MODEL_CLIP_PATH = "plots/Stereo/Winners/SED_0.5__L2_1__huber_1__lr_0.0001__conv__CLIP__use_reconstruction_True/BS_8__ratio_0.2__mid__frozen_0"
KITTI_MODEL_RESNET_PATH = "plots/Stereo/Winners/SED_0.5__L2_1__huber_1__lr_0.0001__conv__Resnet__use_reconstruction_True/BS_8__ratio_0.2__head__frozen_0__seed_300"
KITTI_MODEL_PATH = KITTI_MODEL_CLIP_PATH if MODEL == CLIP_MODEL_NAME else KITTI_MODEL_RESNET_PATH
TRAINED_VIT = None if MODEL==RESNET_MODEL_NAME or USE_REALESTATE or not PRETEXT_TRAIN else "plots/Affine/BS_32__lr_6e-05__train_size_9216__CLIP__alpha_10__conv__original_rotated/model.pth" # This is for when wanting to fine-tune an already trained vit (for example fine-tuning a vit which had been trained on the affine transfomration task)
PRETRAINED_PATH = None # make sure you set GET_OLD_PATH !! 
AVG_EMBEDDINGS = False
USE_CONV = True
NUM_OUTPUT = 8 if USE_RECONSTRUCTION_LAYER else 9
norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device) if MODEL == CLIP_MODEL_NAME else torch.tensor([0.5, 0.5, 0.5]).to(device)
norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device) if MODEL == CLIP_MODEL_NAME else torch.tensor([0.5, 0.5, 0.5]).to(device)



