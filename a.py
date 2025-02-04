from params import *
import os
for p in [FLYING_MODEL_CLIP_PATH, FLYING_MODEL_CLIP_16_PATH, FLYING_MODEL_DINO_PATH, FLYING_MODEL_RESNET_PATH, FLYING_MODEL_EFFIEICNT_PATH]:
    comp1_p = os.path.join("", p)
    if os.path.exists(comp1_p):
        print(f"Found in comp1: {comp1_p}")
    else:
        print(f"Not found in comp1: {comp1_p}")