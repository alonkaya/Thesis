import torch
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

###########################################  OFIR   #################################################################################

# Notes for Ofir:

# Each time you make a change in the code you need to do these steps here on VS code:
# git pull   ->   git add .   ->   git commit -m "."   ->   git push

# When using computer = 1:
#    First do: conda activate alon_env
#     git pull   ->   git add .   ->   git commit -m "."   ->   git push
#    nohup env CUDA_VISIBLE_DEVICES=0 TORCH_USE_CUDA_DSA=1 python Main.py > output_250_i.log 2>&1 &
#    (change i by increasing the number by 1 each time you run a new run)
# My project aviran Main.py (under the command)
# option 1 - output_250_5.log
# option 3 -  output_250_8.log

# When using computer = 2:
#    git pull   ->   git add .   ->   git commit -m "."   ->   git push
#    Run 2 runs on the SAME GPU.
#        For the first run use:
#            gpuQ.py submit -d any -p /home/alonkay/Thesis -e alon_env -c "python Main.py  > output_i.log 2>&1"
#            (change i by increasing the number by 1 each time you run a new run)
#        For the second run on the same GPU:
#            FIRST DO GIT ADD/COMMIT/PUSH FROM VS CODE AND GIT PULL FROM TERMINAL!!!!!
#            First check which GPU is used by the first run by running nvtop. The GPU number is the one under 'DEV' by the user alonkay
#            Then run the following command with the GPU number you found and replace X with that number: 
#            nohup env CUDA_VISIBLE_DEVICES=X python Main.py > output_146_i.log 2>&1 &
#            (change i by increasing the number by 1 each time you run a new run)
# option 1 - output_146_2.log , option 2 - output_146_3.log , option 3 - output_146_3.log  

## To see the runs: nvtop

## To see the list of the queue: gpuQ.py list
## To kill a run: nvtop -> highlight the process -> fn+f9 -> SIGILL (IF SIGILL DOESN'T WORK USE SIGINT) -> Enter

## If after a while you see that when you submit a run it exists after a short time, then it might mean that you are done with 
## this task, so send an image to me with the output you get when running 'cat output_i.log' (replace i with the LATEST number you submitted)

# After finised with options 1 and 2, do option=3

# After finising with EVERYTING, do option=4 with computer=2

###########################################  OFIR   #################################################################################






# Notes for Alon:
# If resnet doesnt look good try to change learning rate
# For RealEstate you can try freezing layers, playing with the learning rate or trying pretrained ViT on affine task (if its better in kitti stereo), or increasing the train size


SEQ_RATIOS = [0.015]     # 3251, 2166, 1082, 540, 405, 269
SEED = [42, 300, 500]
LR = [1e-4]             
BATCH_SIZE = [8]                                                          
PART = ["head", "mid", "tail"]    

RESNET_MODEL_NAME = 'microsoft/resnet-152'
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
MODEL = CLIP_MODEL_NAME 
USE_REALESTATE = False
REALESTATE_SPLIT = False # 50=4632
STEREO = True
RL_TRAIN_NUM = [50]   #  14=1872  #  18=2136  #  20=2368  #  50=6560
INIT_DATA = True 

# TRAINED_VIT = None if MODEL==RESNET_MODEL_NAME or USE_REALESTATE else "plots/Affine/BS_32__lr_6e-05__train_size_9216__CLIP__alpha_10__conv__original_rotated/model.pth" # This is for when wanting to fine-tune an already trained vit (for example fine-tuning a vit which had been trained on the affine transfomration task)
TRAINED_VIT = None
FROZEN_LAYERS = [0] if MODEL==RESNET_MODEL_NAME or USE_REALESTATE else [0, 4, 8]













































### Dataset ###  
RIGHTCAMVAL = False
CROP = 224
RESIZE = 256
AUGMENTATION = True
RANDOM_CROP = True

### STEREO ###
train_seqeunces_stereo = [0,2,3,5] #  10840 images 
val_sequences_stereo =  [6,7,8]    #  3682 images
test_sequences_stereo = [9]        #  1064 images

### RealEstate ###
RL_TEST_NAMES = ["fe2fadf89a84e92a", "f01e8b6f8e10fdd9", "f1ee9dc6135e5307", "a41df4fa06fd391b", "bc0ebb7482f14795", "9bdd34e784c04e3a", "98ebee1c36ecec55"]  # val 656, test 704
JUMP_FRAMES = 6 
RL_TRAIN_SPLIT_RATIO = 0.7

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
# CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
VIT_MODEL_NAME = "google/vit-base-patch32-224-in21k"
PRETRAINED_PATH =  None # make sure you set GET_OLD_PATH !! 
AVG_EMBEDDINGS = False
USE_CONV = True
NUM_OUTPUT = 8 if USE_RECONSTRUCTION_LAYER else 9
norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device) if MODEL == CLIP_MODEL_NAME else torch.tensor([0.5, 0.5, 0.5]).to(device)
norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device) if MODEL == CLIP_MODEL_NAME else torch.tensor([0.5, 0.5, 0.5]).to(device)
