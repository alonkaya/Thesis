from params import *
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTModel, CLIPImageProcessor, CLIPModel
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
            self.pretrained_model = CLIPModel.from_pretrained(pretrained_model_name)

            # Get input dimension for the MLP based on CLIP configuration
            mlp_input_dim = self.pretrained_model.config.projection_dim

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
        self.criterion = nn.MSELoss() if regress else nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.mlp = MLP(mlp_input_dim*3, mlp_hidden_sizes, num_output)


    def forward(self, x_original, x_rotated):
        if self.clip: # If using CLIP
            x_original = self.clip_image_processor(images=x_original, return_tensors="pt", do_resize=False, do_normalize=False, do_center_crop=False, do_rescale=False, do_convert_rgb=False).to(device)
            x_rotated = self.clip_image_processor(images=x_rotated, return_tensors="pt", do_resize=False, do_normalize=False, do_center_crop=False, do_rescale=False, do_convert_rgb=False).to(device)

            original_embeddings = self.pretrained_model.get_image_features(**x_original)
            rotated_embeddings = self.pretrained_model.get_image_features(**x_rotated)

        else: # If using standard ViT
             original_embeddings = self.pretrained_model(x_original).last_hidden_state[:,0,:] # (batch_size, CLS emebdding vector size)
             rotated_embeddings = self.pretrained_model(x_rotated).last_hidden_state[:,0,:]

        cosine_similarity = torch.nn.functional.cosine_similarity(original_embeddings, rotated_embeddings).detach().cpu() # (batch_size)

        # Create another feature embedding of the element-wise mult between the two embedding vectors
        mul_embedding = original_embeddings.mul(rotated_embeddings)

        # Concatenate both original and rotated embedding vectors
        embeddings = torch.cat([rotated_embeddings, original_embeddings, mul_embedding], dim=1)

        # Train MLP on embedding vectors
        output = self.mlp(embeddings)

        return output, cosine_similarity


    def train_model(self, train_loader, val_loader, num_epochs=10):
        # Lists to store training statistics
        root_train_loss = []
        root_val_loss = []
        train_mae = []
        val_mae = []
        cosine_similarities = []
        all_labels = []
        for epoch in range(num_epochs):
            self.train()

            # Lists to store per-batch statistics
            labels = []
            outputs = []

            for original_image, rotated_image, label in train_loader:
                original_image, rotated_image, label = original_image.to(self.device), rotated_image.to(self.device), label.to(self.device)

                # Foward pass
                output, cosine_similarity = self.forward(original_image, rotated_image)

                # Compute loss
                loss = self.criterion(output, label)

                # Compute Backward pass and gradients
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Extend lists with batch statistics
                labels.extend(label.flatten().tolist())
                outputs.extend(output.detach().cpu().flatten().tolist())
                cosine_similarities.extend(cosine_similarity.tolist())

            # Calculate and store root training loss for the epoch
            train_loss_sqrt = torch.sqrt(loss.detach().cpu()).item()
            root_train_loss.append(train_loss_sqrt)

            # Calculate and store mean absolute error for the epoch
            train_mae.append(mean_absolute_error(labels, outputs))

            # Extend list of all labels with current epoch's labels
            all_labels.extend(labels)

            # Validation
            self.eval()
            val_labels = []
            val_outputs = []
            with torch.no_grad():
                for original_image, rotated_image, val_label in val_loader:
                    original_image, rotated_image, val_label = original_image.to(self.device), rotated_image.to(self.device), val_label.to(self.device)

                    val_output, _ = self.forward(original_image, rotated_image)
                    val_loss = self.criterion(val_output, val_label)
                    val_labels.extend(val_label.flatten().tolist())
                    val_outputs.extend(val_output.cpu().flatten().tolist())

                val_mae.append(mean_absolute_error(val_labels, val_outputs))

            # Calculate and store root validation loss for the epoch
            val_loss_sqrt = torch.sqrt(val_loss.detach().cpu()).item()
            root_val_loss.append(val_loss_sqrt)

            print(f'Epoch {epoch+1}/{num_epochs}, Root validation Loss: {val_loss_sqrt} Root training Loss: {train_loss_sqrt} Training MAE: {train_mae[-1]}\n')

        plot_over_epoch(x=range(1, num_epochs + 1), y=root_train_loss, x_label="Epoch", y_label='Root training Loss')
        plot_over_epoch(x=range(1, num_epochs + 1), y=root_val_loss, x_label="Epoch", y_label='Root validation Loss')
        plot_over_epoch(x=range(1, num_epochs + 1), y=train_mae, x_label="Epoch", y_label='Training MAE')
        plot_over_epoch(x=range(1, num_epochs + 1), y=val_mae, x_label="Epoch", y_label='VAlidation MAE')
        plot_over_epoch(x=[angle * angle_range for angle in all_labels], y=cosine_similarities, x_label="Angle degrees", y_label='Cosine similarity', connecting_lines=False)


        # Save
        # torch.save(self.state_dict(), 'vit_mlp_regressor.pth')

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

clip_model_name = "openai/clip-vit-base-patch32"
vit_model_name = "google/vit-base-patch16-224-in21k"

model = ViTMLPRegressor(mlp_hidden_sizes, num_output, pretrained_model_name=clip_model_name, lr=learning_rate, device=device, regress = True, freeze_pretrained_model=False)
model = model.to(device)

model.train_model(train_loader, val_loader, num_epochs=num_epochs)