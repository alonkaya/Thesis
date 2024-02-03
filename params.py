import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_sizes = [1,16,32]  
num_epochs = 2
learning_rate = 1e-4
penalty_coeffs = [1,2]
jump_frames = 2
train_seqeunces = [0,2]
val_sequences = [1,3,4]
penaltize_normalized_options = [True, False]
mlp_hidden_sizes = [512, 256]

clip_model_name = "openai/clip-vit-base-patch32"
vit_model_name = "google/vit-base-patch16-224-in21k"
show_plots = False
use_reconstruction_layer = False
num_output = 8 if use_reconstruction_layer else 9
epipolar_constraint_threshold = 0.15
use_deepf_nocors = False
move_bad_images = False
