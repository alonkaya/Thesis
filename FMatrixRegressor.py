from params import *
from utils import *
from FunMatrix import *
# from deepF_nocors import *
import torch.optim as optim
from transformers import ViTModel, CLIPImageProcessor, CLIPVisionModel
from sklearn.metrics import mean_absolute_error

class FMatrixRegressor(nn.Module):
    def __init__(self, mlp_hidden_sizes, num_output, pretrained_model_name, lr, penalty_coeff, batch_size, penaltize_normalized, freeze_pretrained_model=False):
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
        self.batch_size = batch_size
        self.penaltize_normalized = penaltize_normalized
        self.lr = lr

        # Check if CLIP model is specified
        if pretrained_model_name == "openai/clip-vit-base-patch32":
            self.clip = True

            # Initialize CLIP processor and pretrained model
            self.clip_image_processor = CLIPImageProcessor.from_pretrained(
                pretrained_model_name)
            self.pretrained_model = CLIPVisionModel.from_pretrained(
                pretrained_model_name).to(device)

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

        self.mlp = MLP(mlp_input_dim*7*7*3, mlp_hidden_sizes,
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

            x1_embeddings = self.pretrained_model(**x1).last_hidden_state[:, 1:, :].view(-1,  7*7*768).to(device)
            x2_embeddings = self.pretrained_model(**x2).last_hidden_state[:, 1:, :].view(-1,  7*7*768).to(device)

            # Create another feature embedding of the element-wise mult between the two embedding vectors
            mul_embedding = x1_embeddings.mul(x2_embeddings)

            # Concatenate both original and rotated embedding vectors
            embeddings = torch.cat([x1_embeddings, x2_embeddings, mul_embedding], dim=1)

            # Train MLP on embedding vectors            
            output = self.mlp(embeddings).to(device)

            unnormalized_output = output.view(-1,3,3) if not use_reconstruction_layer else torch.stack([self.get_fmat(x)for x in output])
            
            output = norm_layer(unnormalized_output.view(-1, 9)).view(-1,3,3)
            
            if self.penaltize_normalized:
                penalty = last_sing_value_penalty(output).to(device) if not use_reconstruction_layer else torch.tensor(0).to(device)    
            else:
                penalty = last_sing_value_penalty(unnormalized_output).to(device) if not use_reconstruction_layer else torch.tensor(0).to(device)    
            
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
                unnormalized_output, output, penalty = self.forward(first_image, second_image)

                # Compute loss
                l2_loss = self.L2_loss(output, label)
                loss = l2_loss + self.penalty_coeff*penalty 
                avg_loss += loss.detach()

                # Compute Backward pass and gradients
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Compute train mean epipolar constraint error
                avg_ec_err_truth, avg_ec_err_pred, avg_ec_err_pred_unormalized = get_avg_epipolar_test_errors(
                    first_image.detach(), second_image.detach(), unormalized_label.detach(), output.detach(), unnormalized_output.detach())
                epoch_avg_ec_err_truth += avg_ec_err_truth
                epoch_avg_ec_err_pred += avg_ec_err_pred
                epoch_avg_ec_err_pred_unormalized += avg_ec_err_pred_unormalized

                # Extend lists with batch statistics
                labels = torch.cat((labels, label.detach()), dim=0)
                outputs = torch.cat((outputs, output.detach()), dim=0)

            # Calculate and store mean absolute error for the epoch
            mae = torch.mean(torch.abs(labels - outputs))

            epoch_avg_ec_err_truth, epoch_avg_ec_err_pred, epoch_avg_ec_err_pred_unormalized, avg_loss = (
                v / len(train_loader) for v in (epoch_avg_ec_err_truth, epoch_avg_ec_err_pred, epoch_avg_ec_err_pred_unormalized, avg_loss))

            train_mae.append(mae.cpu().item())
            ec_err_truth.append(epoch_avg_ec_err_truth.cpu().item())
            ec_err_pred.append(epoch_avg_ec_err_pred.cpu().item())
            ec_err_pred_unoramlized.append(epoch_avg_ec_err_pred_unormalized.cpu().item())
            all_train_loss.append(avg_loss.cpu().item())


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
                    val_epoch_avg_ec_err_truth += val_avg_ec_err_truth
                    val_epoch_avg_ec_err_pred += val_avg_ec_err_pred
                    val_epoch_avg_ec_err_pred_unormalized += val_avg_ec_err_pred_unormalized

                    val_outputs = torch.cat((val_outputs, val_output), dim=0)
                    val_labels = torch.cat((val_labels, val_label), dim=0)


                # Calculate and store mean absolute error for the epoch
                mae = torch.mean(torch.abs(val_labels - val_outputs))

                val_epoch_avg_ec_err_truth, val_epoch_avg_ec_err_pred_unormalized, val_epoch_avg_ec_err_pred, epoch_penalty, val_avg_loss = (
                    v / len(val_loader) for v in (val_epoch_avg_ec_err_truth, val_epoch_avg_ec_err_pred_unormalized, val_epoch_avg_ec_err_pred, epoch_penalty, val_avg_loss))

                val_mae.append(mae.cpu().item())
                val_ec_err_truth.append(val_epoch_avg_ec_err_truth.cpu().item())
                val_ec_err_pred.append(val_epoch_avg_ec_err_pred.cpu().item())
                val_ec_err_pred_unormalized.append(val_epoch_avg_ec_err_pred_unormalized.cpu().item())
                all_val_loss.append(val_avg_loss.cpu().item())
                all_penalty.append(epoch_penalty.cpu().item())
            

            if check_nan(all_train_loss[-1], all_val_loss[-1], train_mae[-1], val_mae[-1], ec_err_pred_unoramlized[-1], val_ec_err_pred_unormalized[-1], ec_err_pred[-1],all_penalty[-1]): return
                
            epoch_output = f"""Epoch {epoch+1}/{num_epochs}, Training Loss: {all_train_loss[-1]} Val Loss: {all_val_loss[-1]} 
            Training MAE: {train_mae[-1]} Val mae: {val_mae[-1]} 
            Train epipolar error pred unormalized: {ec_err_pred_unoramlized[-1]} Val epipolar error pred unormalized: {val_ec_err_pred_unormalized[-1]}
            Train epipolar error pred: {ec_err_pred[-1]} Val epipolar error pred: {val_ec_err_pred[-1]} 
            penalty: {all_penalty[-1]}\n"""
            print_and_write(epoch_output)
        
        output = f"""Train ground truth error: {ec_err_truth[-1]} val ground truth error: {val_ec_err_truth[-1]}\n\n\n"""
        print_and_write(output)

        plot_over_epoch(x=range(1, num_epochs + 1), y1=all_train_loss, y2=all_val_loss, 
                        title="Loss", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, learning_rate=self.lr)
        
        plot_over_epoch(x=range(1, num_epochs + 1), y1=train_mae, y2=val_mae, 
                        title="MAE", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, learning_rate=self.lr)
        
        plot_over_epoch(x=range(1, num_epochs + 1), y1=ec_err_pred_unoramlized, y2=val_ec_err_pred_unormalized, 
                        title="Epipolar error unnormalized F", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, learning_rate=self.lr)
        
        plot_over_epoch(x=range(1, num_epochs + 1), y1=ec_err_pred, y2=val_ec_err_pred, 
                        title="Epipolar error F", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, learning_rate=self.lr)
  


    # def get_rotation(self, rx, ry, rz):
    #     # normalize input?
    #     R_x = nn.Parameter(torch.tensor([
    #         [1.,    0.,             0.],
    #         [0.,    torch.cos(rx),    -torch.sin(rx)],
    #         [0.,    torch.sin(rx),     torch.cos(rx)]
    #     ]).to(device))

    #     R_y = nn.Parameter(torch.tensor([
    #         [torch.cos(ry),    0.,    -torch.sin(ry)],
    #         [0.,            1.,     0.],
    #         [torch.sin(ry),    0.,     torch.cos(ry)]
    #     ]).to(device))

    #     R_z = nn.Parameter(torch.tensor([
    #         [torch.cos(rz),    -torch.sin(rz),    0.],
    #         [torch.sin(rz),    torch.cos(rz),     0.],
    #         [0.,            0.,             1.]
    #     ]).to(device))
    #     R = torch.matmul(R_x, torch.matmul(R_y, R_z))
    #     return R

    # def get_inv_intrinsic(self, f):
    #     # TODO: What about the proncipal points?
    #     return nn.Parameter(torch.tensor([
    #         [-1/(f+1e-8),   0.,             0.],
    #         [0.,            -1/(f+1e-8),    0.],
    #         [0.,            0.,             1.]
    #     ]).to(device))

    # def get_translate(self, tx, ty, tz):
    #     return nn.Parameter(torch.tensor([
    #         [0.,  -tz, ty],
    #         [tz,  0,   -tx],
    #         [-ty, tx,  0]
    #     ]).to(device))

    # def get_fmat(self, x):
    #     # F = K2^(-T)*R*[t]x*K1^(-1)
    #     # Note: only need out-dim = 8
    #     R_x = nn.Parameter(torch.tensor([
    #         [1.,    0.,             0.],
    #         [0.,    torch.cos(x[2]),    -torch.sin(x[2])],
    #         [0.,    torch.sin(x[2]),     torch.cos(x[2])]
    #     ]).to(device))

    #     R_y = nn.Parameter(torch.tensor([
    #         [torch.cos(x[3]),    0.,    -torch.sin(x[3])],
    #         [0.,            1.,     0.],
    #         [torch.sin(x[3]),    0.,     torch.cos(x[3])]
    #     ]).to(device))

    #     R_z = nn.Parameter(torch.tensor([
    #         [torch.cos(x[4]),    -torch.sin(x[4]),    0.],
    #         [torch.sin(x[4]),    torch.cos(x[4]),     0.],
    #         [0.,            0.,             1.]
    #     ]).to(device))
    #     R = torch.matmul(R_x, torch.matmul(R_y, R_z))

    #     K1_inv = nn.Parameter(torch.tensor([
    #                 [-1/(x[0]+1e-8),   0.,             0.],
    #                 [0.,            -1/(x[0]+1e-8),    0.],
    #                 [0.,            0.,             1.]
    #             ]).to(device))

    #     K2_inv = nn.Parameter(torch.tensor([
    #                 [-1/(x[1]+1e-8),   0.,             0.],
    #                 [0.,            -1/(x[1]+1e-8),    0.],
    #                 [0.,            0.,             1.]
    #             ]).to(device))

    #     T = nn.Parameter(torch.tensor([
    #                 [0.,  -x[7], x[6]],
    #                 [x[7],  0,   -x[5]],
    #                 [-x[6], x[5],  0]
    #             ]).to(device))

    #     # K1_inv = self.get_inv_intrinsic(x[0])
    #     # K2_inv = self.get_inv_intrinsic(x[1])  # TODO: K2 should be -t not just -1..
    #     # R = self.get_rotation(x[2], x[3], x[4])
    #     # T = self.get_translate(x[5], x[6], x[7])
    #     F = torch.matmul(K2_inv,torch.matmul(R, torch.matmul(T, K1_inv)))

    #     return F


