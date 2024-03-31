from params import *
from utils import *
from FunMatrix import *
import torch.optim as optim
from transformers import ViTModel, CLIPImageProcessor, CLIPVisionModel

class FMatrixRegressor(nn.Module):
    def __init__(self, lr_vit, lr_mlp, penalty_coeff,  penaltize_normalized, 
                 mlp_hidden_sizes=MLP_HIDDEN_DIM, num_output=NUM_OUTPUT, average_embeddings=AVG_EMBEDDINGS, batch_size=BATCH_SIZE, 
                 batchnorm_and_dropout=BN_AND_DO, freeze_pretrained_model=FREEZE_PRETRAINED_MODEL, overfitting=OVERFITTING, augmentation=AUGMENTATION, 
                 pretrained_model_name=MODEL, unfrozen_layers=UNFROZEN_LAYERS, enforce_rank_2=ENFORCE_RANK_2, use_reconstruction=USE_RECONSTRUCTION_LAYER, RE1_coeff=RE1_COEFF):

        """
        Initialize the ViTMLPRegressor model.

        Args:
        - mlp_hidden_sizes (list): List of hidden layer sizes for the MLP.
        - num_output (int): Number of output units in the final layer.
        - pretrained_model_name (str): Name of the pretrained model to use.
        - lr (float): Learning rate for the optimizer.
        - device (str): Device to which the model should be moved (e.g., "cuda" or "cpu").
        - regress (bool): If True, use Mean Squared Error loss; if False, use Cross Entropy Loss.
        - freeze_pretrained_model (bool): If True, freeze the parameters of the pretrained model.
        """

        super(FMatrixRegressor, self).__init__()
        self.to(device)
        self.penalty_coeff = penalty_coeff
        self.RE1_coeff = RE1_coeff
        self.batch_size = batch_size
        self.penaltize_normalized = penaltize_normalized
        self.lr_vit = lr_vit
        self.lr_mlp = lr_mlp
        self.batchnorm_and_dropout = batchnorm_and_dropout
        self.overfitting = overfitting
        self.average_embeddings = average_embeddings
        self.pretrained_model_name = pretrained_model_name
        self.augmentation = augmentation
        self.enforce_rank_2 = enforce_rank_2
        self.use_reconstruction=use_reconstruction

        # Check if CLIP model is specified
        self.clip = True

        # Initialize CLIP processor and pretrained model
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(
            pretrained_model_name)
        self.pretrained_model = CLIPVisionModel.from_pretrained(
            pretrained_model_name).to(device)


        # Get input dimension for the MLP based on ViT configuration
        self.model_hidden_size = self.pretrained_model.config.hidden_size
        if self.average_embeddings:
            mlp_input_shape = 2*self.model_hidden_size
        else:
            mlp_input_shape = 7*7*2*self.model_hidden_size


        self.mlp = MLP(mlp_input_shape, mlp_hidden_sizes,
                       num_output, batchnorm_and_dropout).to(device)
        params = [
            {'params': self.pretrained_model.parameters(), 'lr': lr_vit},  # Lower learning rate for the pre-trained vision transformer
            {'params': self.mlp.parameters(), 'lr': lr_mlp}   # Potentially higher learning rate for the MLP
        ]
        
        self.L2_loss = nn.MSELoss().to(device)
        self.optimizer = optim.Adam(params, lr=lr_vit)


    def get_embeddings(self, x1, x2, predict_t=False):
        processor = self.clip_image_processor_t if predict_t else self.clip_image_processor
        model = self.pretrained_model_t if predict_t else self.pretrained_model
        try:
            x1 = processor(images=x1, return_tensors="pt", do_resize=False, do_normalize=False, do_center_crop=False, do_rescale=False, do_convert_rgb=False)
            x2 = processor(images=x2, return_tensors="pt", do_resize=False, do_normalize=False, do_center_crop=False, do_rescale=False, do_convert_rgb=False)
        except Exception as e:
            print_and_write(f'processor: {e}')
            return
        
        try:
            x1['pixel_values'] = x1['pixel_values'].to(device)
            x2['pixel_values'] = x2['pixel_values'].to(device)
        except Exception as e:
            print_and_write(f'pixel_values to device: {e}')
            print_memory()
            return
        
        try:
            x1_embeddings = model(**x1).last_hidden_state[:, 1:, :].view(-1, 7*7*self.model_hidden_size)
            x2_embeddings = model(**x2).last_hidden_state[:, 1:, :].view(-1, 7*7*self.model_hidden_size) 

        except Exception as e:
            print_and_write(f'clip: {e}')
            print_memory()
            return
        

        if self.average_embeddings:
            try:
                avg_patches = nn.AdaptiveAvgPool2d(1)
                x1_embeddings = avg_patches(x1_embeddings.view(-1, self.model_hidden_size, 7, 7)).view(-1, self.model_hidden_size)
                x2_embeddings = avg_patches(x2_embeddings.view(-1, self.model_hidden_size, 7, 7)).view(-1, self.model_hidden_size)
            except Exception as e: 
                print_and_write(f'avg_patches: {e}')
                return

        embeddings = torch.cat([x1_embeddings, x2_embeddings], dim=1)

        return embeddings

    def forward(self, x1, x2, predict_t=False):
        # Get embeddings from images
        try:
            embeddings = self.get_embeddings(x1, x2, predict_t=predict_t)
        except Exception as e:
            print_and_write(f'get_embeddings: {e}')
            return

        # Apply MLP on embedding vectors
        try:        
            unormalized_output = self.mlp(embeddings).view(-1,3,3) 
        except Exception as e:
            print_and_write(f'mlp: {e}')
            return
        
        # Apply norm layer
        try:            
            output = norm_layer(unormalized_output.view(-1, 9)).view(-1,3,3)
        except Exception as e:
            print_and_write(f'norm_layer: {e}')
            return
        
        try:
            penalty = last_sing_value_penalty(unormalized_output) 
        except Exception as e:
            print_and_write(f'last_sing_value_penalty: {e}')
            return

        return unormalized_output, output, penalty


    def train_model(self, train_loader, val_loader, num_epochs):
        # Lists to store training statistics
        all_train_loss, all_train_loss_t, all_val_loss, train_mae, train_mae_t, val_mae, ec_err_truth, ec_err_pred, ec_err_pred_unoramlized, val_ec_err_truth, \
            val_ec_err_pred, val_ec_err_pred_unormalized, all_penalty = [], [], [], [], [], [], [], [], [], [], [], [], []

        for epoch in range(num_epochs):
            try:
                self.train()
                labels, outputs, Rs, ts = torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)
            except Exception as e:
                print_and_write(f'1 {e}')
                print_memory()
                return
            
            try:
                epoch_avg_ec_err_truth, epoch_avg_ec_err_pred, epoch_avg_ec_err_pred_unormalized, avg_loss, avg_loss_R, avg_loss_t, file_num, epoch_penalty = 0, 0, 0, 0, 0, 0, 0, 0
            except Exception as e:
                print_and_write(f'2 {e}')
                return
            
            for first_image, second_image, label, unormalized_label, K in train_loader:
                try:
                    first_image = first_image.to(device)
                except Exception as e:
                    print_and_write(f'2.1 {e}')
                    print_memory()
                    return
                try:
                    second_image = second_image.to(device)
                except Exception as e:
                    print_and_write(f'2.2 {e}')
                    print_memory()
                    return
                try:
                    label = label.to(device)
                except Exception as e:
                    print_and_write(f'2.3 {e}')
                    print_memory()
                    return
                try:
                    unormalized_label = unormalized_label.to(device)
                except Exception as e:
                    print_and_write(f'2.4 {e}')
                    print_memory()
                    return
                try:
                    K = K.to(device)
                except Exception as e:
                    print_and_write(f'2.5 {e}')
                    print_memory()
                    return
                # Forward pass
                try:
                    unormalized_output, output, penalty = self.forward(first_image, second_image)
                    epoch_penalty = epoch_penalty + penalty

                except Exception as e:
                    print_and_write(f'3 {e}')
                    return

                # try:
                #     # Compute train mean epipolar constraint error
                #     avg_ec_err_truth, avg_ec_err_pred, avg_ec_err_pred_unormalized = get_avg_epipolar_test_errors(
                #         first_image.detach(), second_image.detach(), unormalized_label.detach(), output.detach(), unormalized_output.detach(), epoch, file_num=file_num)
                #     epoch_avg_ec_err_truth = epoch_avg_ec_err_truth + avg_ec_err_truth
                #     epoch_avg_ec_err_pred = epoch_avg_ec_err_pred + avg_ec_err_pred
                #     epoch_avg_ec_err_pred_unormalized = epoch_avg_ec_err_pred_unormalized + avg_ec_err_pred_unormalized
                #
                #     file_num += 1
                # except Exception as e:
                #     print_and_write(f'4 {e}')
                #     return
                try:
                    # Compute loss
                    l2_loss = self.L2_loss(output, label)
                    loss = l2_loss + self.penalty_coeff*penalty
                    avg_loss = avg_loss + loss.detach()

                    # Compute Backward pass and gradients
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Extend lists with batch statistics
                    labels = torch.cat((labels, label.detach()), dim=0)
                    outputs = torch.cat((outputs, output.detach()), dim=0)
                except Exception as e:
                    print_and_write(f'5 {e}')
                    return
            try:
                mae = torch.mean(torch.abs(labels - outputs))

                # epoch_avg_ec_err_truth, epoch_avg_ec_err_pred, epoch_avg_ec_err_pred_unormalized, avg_loss, epoch_penalty = (
                #     v / len(train_loader) for v in (epoch_avg_ec_err_truth, epoch_avg_ec_err_pred, epoch_avg_ec_err_pred_unormalized, avg_loss, epoch_penalty))
                #
                train_mae.append(mae.cpu().item())
                all_train_loss.append(avg_loss.cpu().item())
                # all_penalty.append(epoch_penalty.cpu().item())
                #
                # ec_err_truth.append(epoch_avg_ec_err_truth.cpu().item())
                # ec_err_pred.append(epoch_avg_ec_err_pred.cpu().item())
                # ec_err_pred_unoramlized.append(epoch_avg_ec_err_pred_unormalized.cpu().item())
                
            except Exception as e:
                print_and_write(f'7 {e}')
                return

            try:
                epoch_output = f"""Epoch {epoch+1}/{num_epochs}, Training Loss: {all_train_loss[-1]} 
                Training MAE: {train_mae[-1]}\n"""

                print_and_write(epoch_output)
            except Exception as e:
                print_and_write(f'9 {e}')

        
        output = f"""Train unormalized ground truth error: {np.mean(ec_err_truth)} val unormalized ground truth error: {np.mean(val_ec_err_truth)}\n\n\n"""
        print_and_write(output)
 
        plot_over_epoch(x=range(1, num_epochs + 1), y1=all_train_loss, y2=all_val_loss, 
                        title="Loss" if not self.predict_pose else "Loss R", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, batchnorm_and_dropout=self.batchnorm_and_dropout, 
                        lr_mlp = self.lr_mlp, lr_vit = self.lr_vit, overfitting=self.overfitting, average_embeddings=self.average_embeddings, 
                        model=self.pretrained_model_name, augmentation=self.augmentation, enforce_rank_2=self.enforce_rank_2,
                        use_reconstruction=self.use_reconstruction)
        
        plot_over_epoch(x=range(1, num_epochs + 1), y1=train_mae, y2=val_mae, 
                        title="MAE" if not self.predict_pose else "MAE R", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, batchnorm_and_dropout=self.batchnorm_and_dropout, 
                        lr_mlp = self.lr_mlp, lr_vit = self.lr_vit, overfitting=self.overfitting, average_embeddings=self.average_embeddings, 
                        model=self.pretrained_model_name, augmentation=self.augmentation, enforce_rank_2=self.enforce_rank_2,
                        use_reconstruction=self.use_reconstruction)
        
        # if self.predict_pose:
        #     plot_over_epoch(x=range(1, num_epochs + 1), y1=all_train_loss_t, y2=all_val_loss, 
        #                     title="Loss t", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, batchnorm_and_dropout=self.batchnorm_and_dropout, 
        #                     lr_mlp = self.lr_mlp, lr_vit = self.lr_vit, overfitting=self.overfitting, average_embeddings=self.average_embeddings, 
        #                     model=self.pretrained_model_name, augmentation=self.augmentation, enforce_rank_2=self.enforce_rank_2,
        #                     use_reconstruction=self.use_reconstruction)     
            
        #     plot_over_epoch(x=range(1, num_epochs + 1), y1=train_mae_t, y2=val_mae, 
        #                     title="MAE t", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, batchnorm_and_dropout=self.batchnorm_and_dropout, 
        #                     lr_mlp = self.lr_mlp, lr_vit = self.lr_vit, overfitting=self.overfitting, average_embeddings=self.average_embeddings, 
        #                     model=self.pretrained_model_name, augmentation=self.augmentation, enforce_rank_2=self.enforce_rank_2,
        #                     use_reconstruction=self.use_reconstruction)           
        
        plot_over_epoch(x=range(1, num_epochs + 1), y1=ec_err_pred_unoramlized, y2=val_ec_err_pred_unormalized, 
                        title="Epipolar error unormalized F", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, batchnorm_and_dropout=self.batchnorm_and_dropout, 
                        lr_mlp = self.lr_mlp, lr_vit = self.lr_vit, overfitting=self.overfitting, average_embeddings=self.average_embeddings, 
                        model=self.pretrained_model_name, augmentation=self.augmentation, enforce_rank_2=self.enforce_rank_2,
                        use_reconstruction=self.use_reconstruction)
        
        plot_over_epoch(x=range(1, num_epochs + 1), y1=ec_err_pred, y2=val_ec_err_pred, 
                        title="Epipolar error F", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, batchnorm_and_dropout=self.batchnorm_and_dropout,
                        lr_mlp = self.lr_mlp, lr_vit = self.lr_vit, overfitting=self.overfitting, average_embeddings=self.average_embeddings, 
                        model=self.pretrained_model_name, augmentation=self.augmentation, enforce_rank_2=self.enforce_rank_2,
                        use_reconstruction=self.use_reconstruction)


def print_memory(device_index=0):
    """
    Prints the CUDA memory information for the specified device.

    Parameters:
    - device_index (int): Index of the CUDA device for which the memory information will be printed.
    """
    device = torch.device(f'cuda:{device_index}')  # Adjust device index as per your setup
    print(f"Memory information for device: {torch.cuda.get_device_name(device)}\n")

    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    cached_memory = torch.cuda.memory_reserved(device)
    peak_allocated_memory = torch.cuda.max_memory_allocated(device)
    peak_cached_memory = torch.cuda.max_memory_reserved(device)

    print(f"Total Memory: {total_memory / 1024 ** 3:.2f} GB")
    print(f"Allocated Memory: {allocated_memory / 1024 ** 3:.2f} GB")
    print(f"Cached Memory: {cached_memory / 1024 ** 3:.2f} GB")
    print(f"Peak Allocated Memory: {peak_allocated_memory / 1024 ** 3:.2f} GB")
    print(f"Peak Cached Memory: {peak_cached_memory / 1024 ** 3:.2f} GB\n")

    # Resetting peak memory stats can be useful to understand memory usage over time
    torch.cuda.reset_peak_memory_stats(device)