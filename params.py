import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_of_training_images = 1000
num_of_val_images = 200
learning_rate = 1e-4
mlp_hidden_sizes = [512, 256]
num_epochs = 50
angle_range = 90
shift_x_range = 140
shift_y_range = 140
clip_model_name = "openai/clip-vit-base-patch32"
vit_model_name = "google/vit-base-patch16-224-in21k"
show_plots = False
jump_frames = 2
use_reconstruction_layer = False
num_output = 8 if use_reconstruction_layer else 9
penalty_coeff = 2
epipolar_constraint_threshold = 0.15
batch_size = 1  # If increase batch size beware of zeros in batch!!
use_deepf_nocors = False
move_bad_images = False
train_seqeunces = [0]
val_sequences = [4]