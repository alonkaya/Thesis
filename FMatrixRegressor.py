from params import *
from utils import *
from FunMatrix import *
# from deepF_nocors import *
import torch.optim as optim
from transformers import ViTModel, CLIPImageProcessor, CLIPVisionModel
from sklearn.metrics import mean_absolute_error


class FMatrixRegressor(nn.Module):
    def __init__(self, mlp_hidden_sizes, num_output, pretrained_model_name, lr, freeze_pretrained_model=True):
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

        # Check if CLIP model is specified
        if pretrained_model_name == "openai/clip-vit-base-patch32":
            self.clip = True

            # Initialize CLIP processor and pretrained model
            self.clip_image_processor = CLIPImageProcessor.from_pretrained(
                pretrained_model_name)
            self.pretrained_model = CLIPVisionModel.from_pretrained(
                pretrained_model_name).to(device)
            # self.pretrained_model = CLIPModel.from_pretrained(pretrained_model_name)

            # Get input dimension for the MLP based on CLIP configuration
            mlp_input_dim = self.pretrained_model.config.hidden_size
            # mlp_input_dim = self.pretrained_model.config.projection_dim

        else:
            self.clip = False

            # Initialize ViT pretrained model
            self.pretrained_model = ViTModel.from_pretrained(
                pretrained_model_name).to(device)

            # Get input dimension for the MLP based on ViT configuration
            mlp_input_dim = self.pretrained_model.config.hidden_size

        # Freeze the parameters of the pretrained model if specified
        if freeze_pretrained_model:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        # Choose appropriate loss function based on regress parameter
        self.L2_loss = nn.MSELoss().to(device)
        self.L1_loss = nn.L1Loss().to(device)

        self.mlp = MLP(mlp_input_dim*7*7*2, mlp_hidden_sizes,
                       num_output).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x1, x2):
        if use_deepf_nocors:
            # net = HomographyNet(use_reconstruction_module=False).to(device)

            # output = net.foward(x1, x2).to(device)

            # return output
            ""
        else:
            if self.clip:  # If using CLIP
                x1 = self.clip_image_processor(images=x1, return_tensors="pt", do_resize=False, do_normalize=False,
                                               do_center_crop=False, do_rescale=False, do_convert_rgb=False).to(device)
                x2 = self.clip_image_processor(images=x2, return_tensors="pt", do_resize=False, do_normalize=False,
                                               do_center_crop=False, do_rescale=False, do_convert_rgb=False).to(device)

            x1_embeddings = self.pretrained_model(
                **x1).last_hidden_state[:, :49, :].view(-1, 7*7*768).to(device)
            x2_embeddings = self.pretrained_model(
                **x2).last_hidden_state[:, :49, :].view(-1, 7*7*768).to(device)

            # cosine_similarity = torch.nn.functional.cosine_similarity(x1_embeddings, x2_embeddings).detach().cpu() # (batch_size)

            # Create another feature embedding of the element-wise mult between the two embedding vectors
            # mul_embedding = x1_embeddings.mul(x2_embeddings)

            # Concatenate both original and rotated embedding vectors
            embeddings = torch.cat(
                [x1_embeddings, x2_embeddings], dim=1).to(device)

            # Train MLP on embedding vectors            
            output = self.mlp(embeddings).to(device)

            unnormalized_output = torch.stack([self.get_fmat(x)for x in output]) if use_reconstruction_layer else output.view(-1,3,3)
            
            penalty = torch.tensor(0).to(device) if use_reconstruction_layer else last_sing_value_penalty(unnormalized_output).to(device)

            output = norm_layer(unnormalized_output.view(-1, 9)).view(-1,3,3)

            return unnormalized_output, output, penalty


    def train_model(self, train_loader, val_loader, num_epochs):
        # Lists to store training statistics
        all_train_loss, all_val_loss, train_mae, val_mae, ec_err_truth, ec_err_pred, ec_err_pred_unoramlized, val_ec_err_truth, \
            val_ec_err_pred, val_ec_err_pred_unormalized, all_penalty = [], [], [], [], [], [], [], [], [], [], []

        for epoch in range(num_epochs):
            self.train()
            labels, outputs = torch.tensor([]).to(device), torch.tensor([]).to(device)
            epoch_avg_ec_err_truth, epoch_avg_ec_err_pred, epoch_avg_ec_err_pred_unormalized, avg_loss = 0, 0, 0, 0

            for first_image, second_image, label, unormalized_label in train_loader:
                first_image, second_image, label, unormalized_label = first_image.to(
                    device), second_image.to(device), label.to(device), unormalized_label.to(device)

                # Forward pass
                unnormalized_output, output, penalty = self.forward(
                    first_image, second_image)

                # Compute loss
                l2_loss = self.L2_loss(output, label)
                loss = l2_loss + penalty 
                avg_loss += loss.detach()

                # Compute Backward pass and gradients
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Compute train mean epipolar constraint error
                avg_ec_err_truth, avg_ec_err_pred, avg_ec_err_pred_unormalized = get_avg_epipolar_test_errors(
                    first_image.detach(), second_image.detach(), unormalized_label.detach(), output.detach(), unnormalized_output.detach())
                # avg_ec_err_truth, avg_ec_err_pred, avg_ec_err_pred_unormalized = torch.tensor(0).to(device), torch.tensor(0).to(device), torch.tensor(0).to(device)
                epoch_avg_ec_err_truth += avg_ec_err_truth
                epoch_avg_ec_err_pred += avg_ec_err_pred
                epoch_avg_ec_err_pred_unormalized += avg_ec_err_pred_unormalized

                # Extend lists with batch statistics
                labels = torch.cat((labels, label.detach()), dim=0)
                outputs = torch.cat((outputs, output.detach()), dim=0)

                # cosine_similarities.extend(cosine_similarity.tolist())

            # Calculate and store mean absolute error for the epoch
            mae = torch.mean(torch.abs(labels - outputs))

            epoch_avg_ec_err_truth, epoch_avg_ec_err_pred, epoch_avg_ec_err_pred_unormalized, avg_loss = (
                v / len(train_loader) for v in (epoch_avg_ec_err_truth, epoch_avg_ec_err_pred, epoch_avg_ec_err_pred_unormalized, avg_loss))

            train_mae.append(mae.cpu())
            ec_err_truth.append(epoch_avg_ec_err_truth.cpu())
            ec_err_pred.append(epoch_avg_ec_err_pred.cpu())
            ec_err_pred_unoramlized.append(epoch_avg_ec_err_pred_unormalized.cpu())
            all_train_loss.append(avg_loss.cpu())

            # Extend list of all labels with current epoch's labels for cosine_similarity plot
            # all_labels.extend(labels)

            # Validation
            self.eval()
            val_labels, val_outputs = torch.tensor([]).to(device), torch.tensor([]).to(device)
            val_epoch_avg_ec_err_truth, val_epoch_avg_ec_err_pred, val_epoch_avg_ec_err_pred_unormalized, epoch_penalty, val_avg_loss = 0, 0, 0, 0, 0

            with torch.no_grad():
                for val_first_image, val_second_image, val_label, val_unormalized_label in val_loader:
                    val_first_image, val_second_image, val_label, val_unormalized_label = val_first_image.to(
                        device), val_second_image.to(device), val_label.to(device), val_unormalized_label.to(device)

                    unnormalized_val_output, val_output, penalty = self.forward(
                        val_first_image, val_second_image)
                    epoch_penalty += penalty

                    val_avg_loss += self.L2_loss(val_output, val_label)

                    # Compute val mean epipolar constraint error
                    val_avg_ec_err_truth, val_avg_ec_err_pred, val_avg_ec_err_pred_unormalized = get_avg_epipolar_test_errors(
                        val_first_image, val_second_image, val_unormalized_label, val_output, unnormalized_val_output)
                    # val_avg_ec_err_truth, val_avg_ec_err_pred, val_avg_ec_err_pred_unormalized = torch.tensor(0).to(device), torch.tensor(0).to(device), torch.tensor(0).to(device)
                    val_epoch_avg_ec_err_truth += val_avg_ec_err_truth
                    val_epoch_avg_ec_err_pred += val_avg_ec_err_pred
                    val_epoch_avg_ec_err_pred_unormalized += val_avg_ec_err_pred_unormalized

                    val_outputs = torch.cat((val_outputs, val_output), dim=0)
                    val_labels = torch.cat((val_labels, val_label), dim=0)


                # Calculate and store mean absolute error for the epoch
                mae = torch.mean(torch.abs(val_labels - val_outputs))

                val_epoch_avg_ec_err_truth, val_epoch_avg_ec_err_pred_unormalized, val_epoch_avg_ec_err_pred, epoch_penalty, val_avg_loss = (
                    v / len(val_loader) for v in (val_epoch_avg_ec_err_truth, val_epoch_avg_ec_err_pred_unormalized, val_epoch_avg_ec_err_pred, epoch_penalty, val_avg_loss))

                val_mae.append(mae.cpu())
                val_ec_err_truth.append(val_epoch_avg_ec_err_truth.cpu())
                val_ec_err_pred.append(val_epoch_avg_ec_err_pred.cpu())
                val_ec_err_pred_unormalized.append(val_epoch_avg_ec_err_pred_unormalized.cpu())
                all_val_loss.append(val_avg_loss.cpu())
                all_penalty.append(epoch_penalty.cpu())
                

            # 
            # Train avg epipolar constraint error truth: {epoch_avg_ec_err_truth} Val avg epipolar constraint error truth: {val_epoch_avg_ec_err_truth}\n"""
            epoch_output = f"""Epoch {epoch+1}/{num_epochs}, Training Loss: {all_train_loss[-1]} Val Loss: {all_val_loss[-1]} Training MAE: {train_mae[-1]} Val mae: {val_mae[-1]} penalty: {epoch_penalty}
            Train avg epipolar constraint error pred unormalized: {epoch_avg_ec_err_pred_unormalized} Val avg epipolar constraint error pred unormalized: {val_epoch_avg_ec_err_pred_unormalized}
            Train avg epipolar constraint error pred: {epoch_avg_ec_err_pred} Val avg epipolar constraint error pred:  {val_epoch_avg_ec_err_pred}\n"""

            with open("output.txt", "a") as f:
                f.write(epoch_output)
                print(epoch_output)

        with open("output.txt", "a") as f:
            output = f'Train gorund truth error: {epoch_avg_ec_err_truth} val gorund truth error: {val_epoch_avg_ec_err_truth}\n'
            f.write(output)
            print(output)

        plot_over_epoch(x=range(1, num_epochs + 1), y=all_train_loss,
                        x_label="Epoch", y_label='Training Loss', show=show_plots)
        plot_over_epoch(x=range(1, num_epochs + 1), y=all_val_loss,
                        x_label="Epoch", y_label='Validation Loss', show=show_plots)
        plot_over_epoch(x=range(1, num_epochs + 1), y=train_mae,
                        x_label="Epoch", y_label='Training MAE', show=show_plots)
        plot_over_epoch(x=range(1, num_epochs + 1), y=val_mae,
                        x_label="Epoch", y_label='VAlidation MAE', show=show_plots)
        plot_over_epoch(x=range(1, num_epochs + 1), y=ec_err_pred, x_label="Epoch", y_label='Training epipolar constraint err for pred F', show=show_plots)
        plot_over_epoch(x=range(1, num_epochs + 1), y=ec_err_pred_unoramlized, x_label="Epoch",
                        y_label='Train epipolar constraint err for pred F unormalized', show=show_plots)
        # plot_over_epoch(x=range(1, num_epochs + 1), y=val_ec_err_pred, x_label="Epoch", y_label='Val epipolar constraint err for pred F', show=show_plots)
        plot_over_epoch(x=range(1, num_epochs + 1), y=val_ec_err_pred_unormalized, x_label="Epoch",
                        y_label='Val epipolar constraint err for pred F unormalized', show=show_plots)
        plot_over_epoch(x=range(1, num_epochs + 1), y=all_penalty, x_label="Epoch",
                        y_label='Additional loss penalty for last singular value', show=show_plots)
        # plot_over_epoch(x=[angle * angle_range for angle in all_labels], y=cosine_similarities, x_label="Angle degrees", y_label='Cosine similarity', connecting_lines=False, show=show_plots)

    
    def get_rotation(self, rx, ry, rz):
        # normalize input?
        R_x = nn.Parameter(torch.tensor([
            [1.,    0.,             0.],
            [0.,    torch.cos(rx),    -torch.sin(rx)],
            [0.,    torch.sin(rx),     torch.cos(rx)]
        ]).to(device))

        R_y = nn.Parameter(torch.tensor([
            [torch.cos(ry),    0.,    -torch.sin(ry)],
            [0.,            1.,     0.],
            [torch.sin(ry),    0.,     torch.cos(ry)]
        ]).to(device))

        R_z = nn.Parameter(torch.tensor([
            [torch.cos(rz),    -torch.sin(rz),    0.],
            [torch.sin(rz),    torch.cos(rz),     0.],
            [0.,            0.,             1.]
        ]).to(device))
        R = torch.matmul(R_x, torch.matmul(R_y, R_z))
        return R

    def get_inv_intrinsic(self, f):
        # TODO: What about the proncipal points?
        return nn.Parameter(torch.tensor([
            [-1/(f+1e-8),   0.,             0.],
            [0.,            -1/(f+1e-8),    0.],
            [0.,            0.,             1.]
        ]).to(device))

    def get_translate(self, tx, ty, tz):
        return nn.Parameter(torch.tensor([
            [0.,  -tz, ty],
            [tz,  0,   -tx],
            [-ty, tx,  0]
        ]).to(device))

    def get_fmat(self, x):
        # F = K2^(-T)*R*[t]x*K1^(-1)
        # Note: only need out-dim = 8
        R_x = nn.Parameter(torch.tensor([
            [1.,    0.,             0.],
            [0.,    torch.cos(x[2]),    -torch.sin(x[2])],
            [0.,    torch.sin(x[2]),     torch.cos(x[2])]
        ]).to(device))

        R_y = nn.Parameter(torch.tensor([
            [torch.cos(x[3]),    0.,    -torch.sin(x[3])],
            [0.,            1.,     0.],
            [torch.sin(x[3]),    0.,     torch.cos(x[3])]
        ]).to(device))

        R_z = nn.Parameter(torch.tensor([
            [torch.cos(x[4]),    -torch.sin(x[4]),    0.],
            [torch.sin(x[4]),    torch.cos(x[4]),     0.],
            [0.,            0.,             1.]
        ]).to(device))
        R = torch.matmul(R_x, torch.matmul(R_y, R_z))

        K1_inv = nn.Parameter(torch.tensor([
                    [-1/(x[0]+1e-8),   0.,             0.],
                    [0.,            -1/(x[0]+1e-8),    0.],
                    [0.,            0.,             1.]
                ]).to(device))

        K2_inv = nn.Parameter(torch.tensor([
                    [-1/(x[1]+1e-8),   0.,             0.],
                    [0.,            -1/(x[1]+1e-8),    0.],
                    [0.,            0.,             1.]
                ]).to(device))

        T = nn.Parameter(torch.tensor([
                    [0.,  -x[7], x[6]],
                    [x[7],  0,   -x[5]],
                    [-x[6], x[5],  0]
                ]).to(device))

        # K1_inv = self.get_inv_intrinsic(x[0])
        # K2_inv = self.get_inv_intrinsic(x[1])  # TODO: K2 should be -t not just -1..
        # R = self.get_rotation(x[2], x[3], x[4])
        # T = self.get_translate(x[5], x[6], x[7])
        F = torch.matmul(K2_inv,torch.matmul(R, torch.matmul(T, K1_inv)))

        return F


def get_avg_epipolar_test_errors(first_image, second_image, unormalized_label, output, unormalized_output):
    # Compute mean epipolar constraint error
    U1, S1, V1 = torch.svd(output)
    U2, S2, V2 = torch.svd(unormalized_output)

    S1[:, -1] = 0
    S2[:, -1] = 0

    output = torch.matmul(torch.matmul(U1, torch.diag_embed(S1)), V1.transpose(1, 2))
    unormalized_output = torch.matmul(torch.matmul(U2, torch.diag_embed(S2)), V2.transpose(1, 2))
    
    avg_ec_err_truth, avg_ec_err_pred, avg_ec_err_pred_unormalized = 0, 0, 0
    for img_1, img_2, F_truth, F_pred, F_pred_unormalized in zip(first_image, second_image, unormalized_label, output, unormalized_output):
        avg_ec_err_truth += EpipolarGeometry(img_1,
                                             img_2, F_truth).get_epipolar_err()
        avg_ec_err_pred += EpipolarGeometry(img_1,
                                            img_2, F_pred).get_epipolar_err()
        avg_ec_err_pred_unormalized += EpipolarGeometry(
            img_1, img_2, F_pred_unormalized).get_epipolar_err()

    avg_ec_err_truth, avg_ec_err_pred, avg_ec_err_pred_unormalized = (
        v / len(first_image) for v in (avg_ec_err_truth, avg_ec_err_pred, avg_ec_err_pred_unormalized))

    return avg_ec_err_truth, avg_ec_err_pred, avg_ec_err_pred_unormalized
