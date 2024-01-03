from params import *
from utils import *
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

            # x1_embeddings = self.pretrained_model.get_image_features(**x1)
            # x2_embeddings = self.pretrained_model.get_image_features(**x2)


        else: # If using standard ViT
             x1_embeddings = self.pretrained_model(**x1).last_hidden_state[:,:49,:].view(-1, 7*7*768).to(self.device)
             x2_embeddings = self.pretrained_model(**x2).last_hidden_state[:,:49,:].view(-1, 7*7*768).to(self.device)

        # cosine_similarity = torch.nn.functional.cosine_similarity(x1_embeddings, x2_embeddings).detach().cpu() # (batch_size)

        # Create another feature embedding of the element-wise mult between the two embedding vectors
        # mul_embedding = x1_embeddings.mul(x2_embeddings)

        # Concatenate both original and rotated embedding vectors
        embeddings = torch.cat([x1_embeddings, x2_embeddings], dim=1).to(self.device)

        # Train MLP on embedding vectors
        output = self.mlp(embeddings)

        # Convert 9-vector output to 3x3 F-matrix
        # output = torch.stack([enforce_fundamental_constraints(F_matrix) for F_matrix in output])

        # Apply reconstruction layer
        # output = torch.stack([reconstruction_module(x, self.device) for x in output]).to(self.device)
        output = torch.tensor([
                [output[1],   0.,             0.],
                [0.,            output[0],    0.],
                [0.,            0.,             1.]
            ])
        

        # Apply abs normalization layer
        # output = torch.stack([normalize_F(x) for x in output]).to(self.device)

        return output


    def train_model(self, train_loader, val_loader, num_epochs):
        # Lists to store training statistics
        all_train_loss = []
        all_val_loss = []
        train_mae = []
        val_mae = []
        for epoch in range(num_epochs):
            self.train()

            # Lists to store per-batch statistics
            labels = []
            outputs = []

            for first_image, second_image, label in train_loader:
                first_image, second_image, label = first_image.to(self.device), second_image.to(self.device), label.to(self.device) 
                           
                # Foward pass
                output = self.forward(first_image, second_image)
                
                # Compute loss
                l1_loss = self.L1_loss(output, label)
                l2_loss = self.L2_loss(output, label)
                loss = l1_loss + l2_loss
                
                # Add a term to the loss that penalizes the smallest singular value being far from zero. This complies with the rank-2 constraint
                # loss = add_last_sing_value_penalty(output, loss)

                # Compute Backward pass and gradients
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

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

            # Extend list of all labels with current epoch's labels for cosine_similarity plot
            # all_labels.extend(labels)

            # Validation
            self.eval()
            val_labels = []
            val_outputs = []
            with torch.no_grad():
                for original_image, translated_image, val_label in val_loader:
                    original_image, translated_image, val_label = original_image.to(self.device), translated_image.to(self.device), val_label.to(self.device)
 
                    val_output = self.forward(original_image, translated_image)

                    val_l1_loss = self.L1_loss(val_output, val_label)
                    val_l2_loss = self.L2_loss(val_output, val_label)
                    val_loss = val_l1_loss + val_l2_loss

                    val_outputs.append(val_output.to(self.device))
                    val_labels.append(val_label)

                # Calculate and store mean absolute error for the epoch
                mae = torch.mean(torch.abs(torch.cat(val_labels, dim=0) - torch.cat(val_outputs, dim=0)))
                val_mae.append(mae.cpu())

            # Calculate and store root validation loss for the epoch
            val_loss = val_loss.detach().cpu().item()
            all_val_loss.append(val_loss)

            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss} Val Loss: {val_loss} Training MAE: {train_mae[-1]} Val mae: {val_mae[-1]}\n')

        plot_over_epoch(x=range(1, num_epochs + 1), y=all_train_loss, x_label="Epoch", y_label='Training Loss')
        plot_over_epoch(x=range(1, num_epochs + 1), y=all_val_loss, x_label="Epoch", y_label='Validation Loss')
        plot_over_epoch(x=range(1, num_epochs + 1), y=train_mae, x_label="Epoch", y_label='Training MAE')
        plot_over_epoch(x=range(1, num_epochs + 1), y=val_mae, x_label="Epoch", y_label='VAlidation MAE')
        # plot_over_epoch(x=[angle * angle_range for angle in all_labels], y=cosine_similarities, x_label="Angle degrees", y_label='Cosine similarity', connecting_lines=False)

        # Save
        # torch.save(self.state_dict(), 'vit_mlp_regressor.pth')

