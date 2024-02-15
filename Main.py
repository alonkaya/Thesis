from FMatrixRegressor import FMatrixRegressor
from params import *
from utils import print_and_write, init_main
from Dataset import get_data_loaders
import itertools

if __name__ == "__main__":
    init_main()
        
    # Iterate over each combination
    param_combinations = itertools.product(penalty_coeffs, penaltize_normalized_options, learning_rates_vit, learning_rates_mlp)
    
    for i, (penalty_coeff, penaltize_normalized, lr_vit, lr_mlp) in enumerate(param_combinations):
        
        model = FMatrixRegressor(MLP_HIDDEN_DIM, NUM_OUTPUT, 
                                pretrained_model_name=CLIP_MODEL_NAME, lr_vit=lr_vit, lr_mlp=lr_mlp,
                                penalty_coeff=penalty_coeff, batch_size=BATCH_SIZE, batchnorm_and_dropout=BN_AND_DO,
                                penaltize_normalized=penaltize_normalized, freeze_pretrained_model=False).to(device)

        train_loader, val_loader = get_data_loaders(BATCH_SIZE)
        
        parameters = f"""learning rate vit: {lr_vit}, learning rate mlp: {lr_mlp}, mlp_hidden_sizes: {MLP_HIDDEN_DIM}, jump_frames: {JUMP_FRAMES}, penalty_coeff: {penalty_coeff}, use_reconstruction_layer: {USE_RECONSTRUCTION_LAYER}
batch_size: {BATCH_SIZE}, train_seqeunces: {train_seqeunces}, val_sequences: {val_sequences}, penaltize_normalized: {penaltize_normalized}, RealEstate: {USE_REALESTATE} batchnorm & dropout: {BN_AND_DO}\n\n"""
        print_and_write(parameters)

        model.train_model(train_loader, val_loader, num_epochs=NUM_EPOCHS)
