from params import *
from utils import *
import torch
import torch.optim as optim
from transformers import ViTModel, CLIPImageProcessor, CLIPVisionModel
from sklearn.metrics import mean_absolute_error


class ViTMLPRegressor(nn.Module):
    def __init__(self, mlp_hidden_sizes, num_output, pretrained_model_name, lr, device, regress=True, freeze_pretrained_model=True):
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

        super(ViTMLPRegressor, self).__init__()
        self.device = device
        self.to(device)

        # Check if CLIP model is specified
        if pretrained_model_name == "openai/clip-vit-base-patch32":
            self.clip = True

            # Initialize CLIP processor and pretrained model
            self.clip_image_processor = CLIPImageProcessor.from_pretrained(pretrained_model_name)
            self.pretrained_model = CLIPVisionModel.from_pretrained(pretrained_model_name).to(device)

        else:
            self.clip = False

            # Initialize ViT pretrained model
            self.pretrained_model = ViTModel.from_pretrained(pretrained_model_name).to(self.device)

        # Get input dimension for the MLP based on ViT configuration
        self.hidden_size = self.pretrained_model.config.hidden_size
        mlp_input_dim = self.hidden_size
        if embedding_tokens == 2:
            mlp_input_dim = self.hidden_size * 2
        elif embedding_tokens == 3:
            mlp_input_dim = self.hidden_size * 3

        # Freeze the parameters of the pretrained model if specified
        if freeze_pretrained_model:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        # Choose appropriate loss function based on regress parameter
        self.criterion = nn.MSELoss() if regress else nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.mlp = MLP(mlp_input_dim, mlp_hidden_sizes, num_output).to(self.device)


    def forward(self, x_original, x_rotated):
        if self.clip: # If using CLIP
            x_original = self.clip_image_processor(images=x_original, return_tensors="pt", do_resize=False, do_normalize=False, 
                                                   do_center_crop=False, do_rescale=False, do_convert_rgb=False).to(self.device)
            x_rotated = self.clip_image_processor(images=x_rotated, return_tensors="pt", do_resize=False, do_normalize=False, 
                                                  do_center_crop=False, do_rescale=False, do_convert_rgb=False).to(self.device)

        original_embeddings = self.pretrained_model(x_original).last_hidden_state[:,0,:].view(-1,self.hidden_size).to(self.device)
        rotated_embeddings = self.pretrained_model(x_original).last_hidden_state[:,0,:].view(-1,self.hidden_size).to(self.device)

        cosine_similarity = torch.nn.functional.cosine_similarity(original_embeddings, rotated_embeddings) # (batch_size)

         # Create another feature embedding of the element-wise mult between the two embedding vectors
        mul_embedding = original_embeddings.mul(rotated_embeddings)

        if embedding_tokens == 1:
            embeddings = rotated_embeddings
        elif embedding_tokens == 2:
            embeddings = torch.cat([original_embeddings, rotated_embeddings], dim=1)
        elif embedding_tokens == 3:
            embeddings = torch.cat([original_embeddings, rotated_embeddings, mul_embedding], dim=1)

        # Train MLP on embedding vectors
        output = self.mlp(embeddings)

        return output, cosine_similarity


    def train_model(self, train_loader, val_loader, num_epochs=10):
        # Lists to store training statistics
        all_train_loss = []
        all_val_loss = []
        train_mae = []
        val_mae = []
        cosine_similarities = []
        all_angles = []        
        for epoch in range(num_epochs):
            self.train()
            labels = []
            outputs = []

            for original_image, translated_image, label in train_loader:
                original_image, translated_image, label = original_image.to(self.device), translated_image.to(self.device), label.to(self.device)

                # Foward pass
                output, cosine_similarity = self.forward(original_image, translated_image)

                # Compute loss
                loss = self.criterion(output, label)

                # Compute Backward pass and gradients
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Extend lists with batch statistics
                labels.append(label.detach().cpu())
                outputs.append(output.detach().cpu())
                
                cosine_similarities.extend(cosine_similarity.detach().cpu().tolist())
                all_angles.extend(label[:,0].detach().cpu().tolist())

            # Calculate and store root training loss for the epoch
            all_train_loss.append(loss.detach().cpu().item())

            # Calculate and store mean absolute error for the epoch
            mae = torch.mean(torch.abs(torch.cat(labels, dim=0) - torch.cat(outputs, dim=0)))
            train_mae.append(mae)

            # Validation
            self.eval()
            val_labels = []
            val_outputs = []
            with torch.no_grad():
                for original_image, translated_image, val_label in val_loader:
                    original_image, translated_image, val_label = original_image.to(self.device), translated_image.to(self.device), val_label.to(self.device)
 
                    val_output, _ = self.forward(original_image, translated_image)
                    val_loss = self.criterion(val_output, val_label)

                    val_outputs.append(val_output.cpu())
                    val_labels.append(val_label.cpu())
                
                # Calculate and store root validation loss for the epoch
                all_val_loss.append(val_loss.cpu().item())

                # Calculate and store mean absolute error for the epoch
                mae = torch.mean(torch.abs(torch.cat(val_labels, dim=0) - torch.cat(val_outputs, dim=0)))
                val_mae.append(mae.cpu())

            print(f"""Epoch {epoch+1}/{num_epochs}, Training Loss: {all_train_loss[-1]} Val Loss: {all_val_loss[-1]}
                   Training MAE: {train_mae[-1]} Val mae: {val_mae[-1]}\n""")

        plot_over_epoch(x=range(1, num_epochs + 1), y=all_train_loss, x_label="Epoch", y_label='Training Loss')
        plot_over_epoch(x=range(1, num_epochs + 1), y=all_val_loss, x_label="Epoch", y_label='Validation Loss')
        plot_over_epoch(x=range(1, num_epochs + 1), y=train_mae, x_label="Epoch", y_label='Training MAE')
        plot_over_epoch(x=range(1, num_epochs + 1), y=val_mae, x_label="Epoch", y_label='VAlidation MAE')
        plot_over_epoch(x=[angle * angle_range for angle in all_angles], y=cosine_similarities, x_label="Angle degrees", y_label='Cosine similarity', connecting_lines=False)