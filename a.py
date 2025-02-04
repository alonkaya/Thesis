from params import *
import os
comp1 = "/mnt_hdd15tb/alonkay/Thesis/"
comp0 = "/mnt/sda2/Alon/"

for p in [FLYING_MODEL_CLIP_PATH, FLYING_MODEL_CLIP_16_PATH, FLYING_MODEL_DINO_PATH, FLYING_MODEL_RESNET_PATH, FLYING_MODEL_EFFIEICNT_PATH]:
    model_p = os.path.join(comp0, p)
    if os.path.exists(model_p):
        print(f"Found in comp1: {model_p}")
    else:
        print(f"Not found in comp1: {model_p}")