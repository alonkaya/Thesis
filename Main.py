from FMatrixRegressor import FMatrixRegressor
from params import *
from utils import print_and_write, init_main
from Dataset import get_data_loaders
import itertools
from a import * 

if __name__ == "__main__":
    init_main()
        
    # Iterate over each combination
    param_combinations = itertools.product(penalty_coeffs, penaltize_normalized_options, learning_rates_vit, learning_rates_mlp)
    
    for i, (penalty_coeff, penaltize_normalized, lr_vit, lr_mlp) in enumerate(param_combinations):
        model = FMatrixRegressor(lr_vit=lr_vit, 
                                 lr_mlp=lr_mlp,
                                 penalty_coeff=penalty_coeff,
                                 penaltize_normalized=penaltize_normalized, 
                                ).to(device)

        # train_loader, val_loader = data_for_checking_overfit(BATCH_SIZE, CUSTOMDATASET_TYPE)
        train_loader, val_loader = data_with_one_sequence(BATCH_SIZE, CUSTOMDATASET_TYPE)
        
        parameters = f"""learning rate vit: {lr_vit}, learning rate mlp: {lr_mlp}, mlp_hidden_sizes: {MLP_HIDDEN_DIM}, jump_frames: {JUMP_FRAMES}, penalty_coeff: {penalty_coeff}, use_reconstruction_layer: {USE_RECONSTRUCTION_LAYER}
batch_size: {BATCH_SIZE}, train_seqeunces: {train_seqeunces}, val_sequences: {val_sequences}, penaltize_normalized: {penaltize_normalized}, RealEstate: {USE_REALESTATE}, batchnorm & dropout: {BN_AND_DO}, 
average embeddings: {AVG_EMBEDDINGS}, customdataset type: {CUSTOMDATASET_TYPE}, model: {MODEL}, augmentation: {AUGMENTATION}\n\n"""
        print_and_write(parameters)

        model.train_model(train_loader, val_loader, num_epochs=NUM_EPOCHS)
