from params import *
from utils import *
from FunMatrix import check_epipolar_constraint
from Dataset import train_loader, val_loader
import torch.optim as optim
from transformers import ViTModel, CLIPImageProcessor, CLIPModel, CLIPVisionModel
from sklearn.metrics import mean_absolute_error

class FMatrixRegressor(nn.Module):
    def __init__(self, mlp_hidden_sizes, num_output, pretrained_model_name, lr, device, freeze_pretrained_model=True):
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
        self.device = device
        self.to(device)

        # Check if CLIP model is specified
        if pretrained_model_name == "openai/clip-vit-base-patch32":
            self.clip = True

            # Initialize CLIP processor and pretrained model
            self.clip_image_processor = CLIPImageProcessor.from_pretrained(pretrained_model_name)
            self.pretrained_model = CLIPVisionModel.from_pretrained(pretrained_model_name)
            # self.pretrained_model = CLIPModel.from_pretrained(pretrained_model_name)

            # Get input dimension for the MLP based on CLIP configuration
            mlp_input_dim = self.pretrained_model.config.hidden_size
            # mlp_input_dim = self.pretrained_model.config.projection_dim

        else:
            self.clip = False

            # Initialize ViT pretrained model
            self.pretrained_model = ViTModel.from_pretrained(pretrained_model_name)

            # Get input dimension for the MLP based on ViT configuration
            mlp_input_dim = self.pretrained_model.config.hidden_size

        # Freeze the parameters of the pretrained model if specified
        if freeze_pretrained_model:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        # Choose appropriate loss function based on regress parameter
        self.L2_loss = nn.MSELoss()
        self.L1_loss = nn.L1Loss()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.mlp = MLP(mlp_input_dim*7*7*2, mlp_hidden_sizes, num_output)


    def forward(self, x1, x2):
        if self.clip: # If using CLIP
            x1 = self.clip_image_processor(images=x1, return_tensors="pt", do_resize=False, do_normalize=False, do_center_crop=False, do_rescale=False, do_convert_rgb=False).to(self.device)
            x2 = self.clip_image_processor(images=x2, return_tensors="pt", do_resize=False, do_normalize=False, do_center_crop=False, do_rescale=False, do_convert_rgb=False).to(self.device)

            x1_embeddings = self.pretrained_model(**x1).last_hidden_state[:,:49,:].view(-1, 7*7*768).to(self.device)
            x2_embeddings = self.pretrained_model(**x2).last_hidden_state[:,:49,:].view(-1, 7*7*768).to(self.device)

        else: # If using standard ViT
             x1_embeddings = self.pretrained_model(**x1).last_hidden_state[:,:49,:].view(-1, 7*7*768).to(self.device)
             x2_embeddings = self.pretrained_model(**x2).last_hidden_state[:,:49,:].view(-1, 7*7*768).to(self.device)

        # cosine_similarity = torch.nn.functional.cosine_similarity(x1_embeddings, x2_embeddings).detach().cpu() # (batch_size)

        # Create another feature embedding of the element-wise mult between the two embedding vectors
        # mul_embedding = x1_embeddings.mul(x2_embeddings)

        # Concatenate both original and rotated embedding vectors
        embeddings = torch.cat([x1_embeddings, x2_embeddings], dim=1).to(self.device)

        # Train MLP on embedding vectors
        output = self.mlp(embeddings).view(-1,3,3)


        if add_penalty_loss:
            # Compute penalty for last singular value 
            penalty = last_sing_value_penalty(output).to(self.device)
           
            # Apply L2 norm on top of L1 norm 
            output = torch.stack([normalize_L2(normalize_L1(x)) for x in output]).to(self.device)

        elif enforce_fundamental_constraint:
            # Convert 9-vector output to 3x3 rank-2 F-matrix
            output = torch.stack([enforce_fundamental_constraints(x) for x in output]).to(self.device)

            # Apply L2 norm on top of L1 norm 
            output = torch.stack([normalize_L2(normalize_L1(x)) for x in output]).to(self.device)
        
        else:
            # Apply reconstruction layer to 8-vector output
            output = torch.stack([reconstruction_module(x) for x in output]).to(self.device)        

            # Apply max normalization layer
            output = torch.stack([normalize_max(x) for x in output]).to(self.device)
        
        return output, penalty


    def train_model(self, train_loader, val_loader, num_epochs):
        # Lists to store training statistics
        all_train_loss = []
        all_val_loss = []
        train_mae = []
        val_mae = []
        train_all_avg_ec_err = []
        val_all_avg_ec_err = []
        for epoch in range(num_epochs):
            self.train()

            # Lists to store per-batch statistics
            labels = []
            outputs = []
            epoch_avg_ec_err = 0
            for first_image, second_image, label in train_loader:
                first_image, second_image, label = first_image.to(self.device), second_image.to(self.device), label.to(self.device) 
                           
                # Foward pass
                output, penalty = self.forward(first_image, second_image)
                
                # Compute loss
                l1_loss = self.L1_loss(output, label)
                l2_loss = self.L2_loss(output, label)
                loss = l2_loss + penalty_coeff*penalty

                # Compute Backward pass and gradients
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Compute train mean epipolar constraint error 
                avg_ec_err = 0
                for img_1, img_2, F in zip(first_image, second_image, label):
                    avg_ec_err += check_epipolar_constraint(img_1.detach().cpu(), img_2.detach().cpu(), F.detach().cpu())
                avg_ec_err /= len(first_image)
                epoch_avg_ec_err += avg_ec_err

                # Extend lists with batch statistics
                labels.append(label)
                outputs.append(output.detach().cpu().to(self.device))
                # cosine_similarities.extend(cosine_similarity.tolist())
      
            # Calculate and store root training loss for the epoch
            train_loss = loss.detach().cpu().item()
            all_train_loss.append(train_loss)

            # Calculate and store mean absolute error for the epoch
            mae = torch.mean(torch.abs(torch.cat(labels, dim=0) - torch.cat(outputs, dim=0)))
            train_mae.append(mae.cpu())
            
            epoch_avg_ec_err /= len(train_loader)
            train_all_avg_ec_err.append(epoch_avg_ec_err)
            # Extend list of all labels with current epoch's labels for cosine_similarity plot
            # all_labels.extend(labels)

            # Validation
            self.eval()
            val_labels = []
            val_outputs = []
            val_epoch_avg_ec_err = 0
            total_penalty = 0
            with torch.no_grad():
                for original_image, translated_image, val_label in val_loader:
                    original_image, translated_image, val_label = original_image.to(self.device), translated_image.to(self.device), val_label.to(self.device)
 
                    val_output, penalty = self.forward(original_image, translated_image)
                    total_penalty += penalty
                    val_l1_loss = self.L1_loss(val_output, val_label)
                    val_l2_loss = self.L2_loss(val_output, val_label)
                    val_loss = val_l2_loss 

                    # Compute val mean epipolar constraint error 
                    val_avg_ec_err = 0
                    for img_1, img_2, F in zip(first_image, second_image, label):
                        val_avg_ec_err += check_epipolar_constraint(img_1.detach().cpu(), img_2.detach().cpu(), F.detach().cpu())
                    val_avg_ec_err /= len(first_image)
                    val_epoch_avg_ec_err += val_avg_ec_err

                    val_outputs.append(val_output.to(self.device))
                    val_labels.append(val_label)

                # Calculate and store mean absolute error for the epoch
                mae = torch.mean(torch.abs(torch.cat(val_labels, dim=0) - torch.cat(val_outputs, dim=0)))
                val_mae.append(mae.cpu())

                val_epoch_avg_ec_err /= len(val_loader)
                val_all_avg_ec_err.append(val_epoch_avg_ec_err)

                total_penalty /= len(val_loader)
            # Calculate and store root validation loss for the epoch
            val_loss = val_loss.detach().cpu().item()
            all_val_loss.append(val_loss)

            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss} Val Loss: {val_loss} Training MAE: {train_mae[-1]} Val mae: {val_mae[-1]} Train avg epipolar constraint error: {epoch_avg_ec_err} Val avg epipolar constraint error: {val_epoch_avg_ec_err} penalty: {total_penalty}\n')

        plot_over_epoch(x=range(1, num_epochs + 1), y=all_train_loss, x_label="Epoch", y_label='Training Loss')
        plot_over_epoch(x=range(1, num_epochs + 1), y=all_val_loss, x_label="Epoch", y_label='Validation Loss')
        plot_over_epoch(x=range(1, num_epochs + 1), y=train_mae, x_label="Epoch", y_label='Training MAE')
        plot_over_epoch(x=range(1, num_epochs + 1), y=val_mae, x_label="Epoch", y_label='VAlidation MAE')
        plot_over_epoch(x=range(1, num_epochs + 1), y=train_all_avg_ec_err, x_label="Epoch", y_label='Training mean epipolar constraint err')
        plot_over_epoch(x=range(1, num_epochs + 1), y=val_all_avg_ec_err, x_label="Epoch", y_label='Valdation mean epipolar constraint err')
        # plot_over_epoch(x=[angle * angle_range for angle in all_labels], y=cosine_similarities, x_label="Angle degrees", y_label='Cosine similarity', connecting_lines=False)

        # Save
        # torch.save(self.state_dict(), 'vit_mlp_regressor.pth')

