from params import *
from utils import *
from FunMatrix import *
# from deepF_nocors import *
import torch.optim as optim
from transformers import ViTModel, CLIPImageProcessor, CLIPVisionModel

class FMatrixRegressor(nn.Module):
    def __init__(self, lr_vit, lr_mlp, penalty_coeff,  penaltize_normalized, 
                 mlp_hidden_sizes=MLP_HIDDEN_DIM, num_output=NUM_OUTPUT, average_embeddings=AVG_EMBEDDINGS, batch_size=BATCH_SIZE, 
                 batchnorm_and_dropout=BN_AND_DO, freeze_pretrained_model=FREEZE_PRETRAINED_MODEL, overfitting=OVERFITTING, augmentation=AUGMENTATION, 
                 pretrained_model_name=MODEL, unfrozen_layers=UNFROZEN_LAYERS, enforce_rank_2=ENFORCE_RANK_2, predict_pose=PREDICT_POSE, use_reconstruction=USE_RECONSTRUCTION_LAYER, RE1_coeff=RE1_COEFF):

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
        self.predict_pose = predict_pose

        # Check if CLIP model is specified
        if pretrained_model_name == "openai/clip-vit-base-patch32":
            self.clip = True

            # Initialize CLIP processor and pretrained model
            self.clip_image_processor = CLIPImageProcessor.from_pretrained(
                pretrained_model_name)
            self.pretrained_model = CLIPVisionModel.from_pretrained(
                pretrained_model_name).to(device)
            
            if self.predict_pose:
                self.clip_image_processor_t = CLIPImageProcessor.from_pretrained(
                    pretrained_model_name)
                self.pretrained_model_t = CLIPVisionModel.from_pretrained(
                    pretrained_model_name).to(device)
                

        else:
            self.clip = False

            # Initialize ViT pretrained model
            self.pretrained_model = ViTModel.from_pretrained(
                pretrained_model_name).to(device)

        # Freeze the parameters of the pretrained model if specified
        # if freeze_pretrained_model:
        #     for param in self.pretrained_model.parameters():
        #         param.requires_grad = False
        # print(len(self.pretrained_model.encoder.layer))
        # for layer in self.pretrained_model.encoder.layer[len(self.pretrained_model.encoder.layer)-unfrozen_layers:]:
        #     for param in layer.parameters():
        #         param.requires_grad = True

        # Get input dimension for the MLP based on ViT configuration
        self.model_hidden_size = self.pretrained_model.config.hidden_size
        if self.average_embeddings:
            mlp_input_shape = 2*self.model_hidden_size
        else:
            mlp_input_shape = 7*7*2*self.model_hidden_size

        if GROUP_CONV["use"]:
            mlp_input_shape //= 3

        self.mlp = MLP(mlp_input_shape, mlp_hidden_sizes,
                       num_output, batchnorm_and_dropout).to(device)
        params = [
            {'params': self.pretrained_model.parameters(), 'lr': lr_vit},  # Lower learning rate for the pre-trained vision transformer
            {'params': self.mlp.parameters(), 'lr': lr_mlp}   # Potentially higher learning rate for the MLP
        ]

        if self.predict_pose:
            self.t_mlp = MLP(mlp_input_shape, mlp_hidden_sizes,
                        3, batchnorm_and_dropout).to(device)
            params_t = [
                    {'params': self.pretrained_model_t.parameters(), 'lr': lr_vit},  # Lower learning rate for the pre-trained vision transformer
                    {'params': self.t_mlp.parameters(), 'lr': lr_mlp}   # Potentially higher learning rate for the MLP
                ]  
        
        self.L2_loss = nn.MSELoss().to(device)
        self.optimizer = optim.Adam(params, lr=lr_vit)

        if self.predict_pose:
            self.L2_loss_t = nn.MSELoss().to(device)
            self.optimizer_t = optim.Adam(params_t, lr=lr_mlp)

    def get_embeddings(self, x1, x2, predict_t=False):
        if self.clip:  
            try:
                if predict_t:
                    x1 = self.clip_image_processor_t(images=x1, return_tensors="pt", do_resize=False, do_normalize=False,
                                                do_center_crop=False, do_rescale=False, do_convert_rgb=False).to(device)
                    x2 = self.clip_image_processor_t(images=x2, return_tensors="pt", do_resize=False, do_normalize=False,
                                            do_center_crop=False, do_rescale=False, do_convert_rgb=False).to(device)
                else:
                    x1 = self.clip_image_processor(images=x1, return_tensors="pt", do_resize=False, do_normalize=False,
                                                do_center_crop=False, do_rescale=False, do_convert_rgb=False).to(device)
                    x2 = self.clip_image_processor(images=x2, return_tensors="pt", do_resize=False, do_normalize=False,
                                            do_center_crop=False, do_rescale=False, do_convert_rgb=False).to(device)
                if predict_t:
                    x1_embeddings = self.pretrained_model_t(**x1).last_hidden_state[:, 1:, :].view(-1, 7*7*self.model_hidden_size).to(device)
                    x2_embeddings = self.pretrained_model_t(**x2).last_hidden_state[:, 1:, :].view(-1, 7*7*self.model_hidden_size).to(device)                
                else:
                    x1_embeddings = self.pretrained_model(**x1).last_hidden_state[:, 1:, :].view(-1, 7*7*self.model_hidden_size).to(device)
                    x2_embeddings = self.pretrained_model(**x2).last_hidden_state[:, 1:, :].view(-1, 7*7*self.model_hidden_size).to(device)                
            except Exception as e:
                print_and_write(f'clip: {e}')
        else:
            x1_embeddings = self.pretrained_model(x1).last_hidden_state[:, 1:, :].view(-1,  7*7*self.model_hidden_size).to(device)
            x2_embeddings = self.pretrained_model(x2).last_hidden_state[:, 1:, :].view(-1,  7*7*self.model_hidden_size).to(device)

        if self.average_embeddings:
            try:
                avg_patches = nn.AdaptiveAvgPool2d(1)
                x1_embeddings = avg_patches(x1_embeddings.view(-1, self.model_hidden_size, 7, 7)).view(-1, self.model_hidden_size)
                x2_embeddings = avg_patches(x2_embeddings.view(-1, self.model_hidden_size, 7, 7)).view(-1, self.model_hidden_size)
            except Exception as e: 
                print_and_write(f'avg_patches: {e}')

        if GROUP_CONV["use"]:
            grouped_conv_layer = GroupedConvolution(in_channels=self.model_hidden_size,   # Total input channels
                                    out_channels=GROUP_CONV["out_channels"],  # Total output channels you want
                                    kernel_size=3,
                                    padding=1,
                                    groups=GROUP_CONV["num_groups"])
            x1_embeddings = grouped_conv_layer(x1_embeddings.unsqueeze(2).unsqueeze(3)).view(-1, self.model_hidden_size//3)
            x2_embeddings = grouped_conv_layer(x2_embeddings.unsqueeze(2).unsqueeze(3)).view(-1, self.model_hidden_size//3)

        # Create another feature embedding of the element-wise mult between the two embedding vectors
        # mul_embedding = x1_embeddings.mul(x2_embeddings)

        embeddings = torch.cat([x1_embeddings, x2_embeddings], dim=1)

        return embeddings

    def forward(self, x1, x2, predict_t=False):
        # If deepF_nocors
        # net = HomographyNet(use_reconstruction_module=False).to(device)
        # output = net.foward(x1, x2).to(device)
        # return output

        try:
            embeddings = self.get_embeddings(x1, x2, predict_t=predict_t)
        except Exception as e:
            print_and_write(f'get_embeddings: {e}')
        try:
            # Train MLP on embedding vectors            
            unormalized_output = self.mlp(embeddings).view(-1,3,3).to(device) if not predict_t else self.t_mlp(embeddings).view(-1,3,1).to(device)
        except Exception as e:
            print_and_write(f'mlp: {e}')

        try:            
            output = norm_layer(unormalized_output.view(-1, 9)).view(-1,3,3) if not predict_t else norm_layer(unormalized_output.view(-1, 3), predict_t=True).view(-1,3,1)

            penalty = last_sing_value_penalty(unormalized_output).to(device) if not self.predict_pose else 0
        except Exception as e:
            print_and_write(f'last_sing_value_penalty: {e}')

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
            try:
                epoch_avg_ec_err_truth, epoch_avg_ec_err_pred, epoch_avg_ec_err_pred_unormalized, avg_loss, avg_loss_R, avg_loss_t, file_num, epoch_penalty = 0, 0, 0, 0, 0, 0, 0, 0
            except Exception as e:
                print_and_write(f'2 {e}')
            for first_image, second_image, label, unormalized_label, K in train_loader:
                try:
                    first_image, second_image, label, unormalized_label, K = first_image.to(device), second_image.to(device), label.to(device), unormalized_label.to(device), K.to(device)
                except Exception as e:
                    print_and_write(f'2.5 {e}')
                # Forward pass
                try:
                    if self.predict_pose:
                        unormalized_R, R, _ = self.forward(first_image, second_image, predict_t=False)
                        unormalized_t, t, _ = self.forward(first_image, second_image, predict_t=True)

                        # This is for the epipolar test error computation:
                        unormalized_pose = torch.cat((unormalized_R.detach(), unormalized_t.detach().view(-1, 3, 1)), dim=-1)
                        pose = torch.cat((R.detach(), t.detach().view(-1, 3, 1)), dim=-1)
                        # output = norm_layer(unormalized_output.view(-1, 9)).view(-1,3,3)

                        unormalized_output = pose_to_F(unormalized_pose, K[0])
                        output = pose_to_F(pose, K[0])
                        unormalized_label = pose_to_F(label, K[0]) # notice this is actually normalized label!
                        
                    else:
                        unormalized_output, output, penalty = self.forward(first_image, second_image)
                        epoch_penalty = epoch_penalty + penalty

                except Exception as e:
                    print_and_write(f'3 {e}')

                try:
                    # Compute train mean epipolar constraint error
                    avg_ec_err_truth, avg_ec_err_pred, avg_ec_err_pred_unormalized = get_avg_epipolar_test_errors(
                        first_image.detach(), second_image.detach(), unormalized_label.detach(), output.detach(), unormalized_output.detach(), epoch, file_num=file_num)
                    epoch_avg_ec_err_truth = epoch_avg_ec_err_truth + avg_ec_err_truth
                    epoch_avg_ec_err_pred = epoch_avg_ec_err_pred + avg_ec_err_pred
                    epoch_avg_ec_err_pred_unormalized = epoch_avg_ec_err_pred_unormalized + avg_ec_err_pred_unormalized

                    file_num += 1
                except Exception as e:
                    print_and_write(f'4 {e}')
                try:
                    if self.predict_pose:
                        loss_R = self.L2_loss(R, label[:, :, :3])
                        avg_loss_R += loss_R.detach()

                        loss_t = self.L2_loss_t(t, label[:, :, 3].view(-1,3,1))
                        avg_loss_t += loss_t.detach()   

                        self.optimizer.zero_grad()
                        loss_R.backward()
                        self.optimizer.step()                     

                        self.optimizer_t.zero_grad()
                        loss_t.backward()
                        self.optimizer_t.step()

                        # Extend lists with batch statistics
                        labels = torch.cat((labels, label.detach()), dim=0)
                        Rs = torch.cat((Rs, R.detach()), dim=0)
                        ts = torch.cat((ts, t.detach()), dim=0)

                    else:
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
            try:
                # Calculate and store mean absolute error for the epoch
                if self.predict_pose:
                    mae_R = torch.mean(torch.abs(labels[:, :, :3] - Rs))
                    mae_t = torch.mean(torch.abs(label[:, :, 3].view(-1,3,1) - ts))

                    epoch_avg_ec_err_truth, epoch_avg_ec_err_pred, epoch_avg_ec_err_pred_unormalized, avg_loss_R, avg_loss_t = (
                        v / len(train_loader) for v in (epoch_avg_ec_err_truth, epoch_avg_ec_err_pred, epoch_avg_ec_err_pred_unormalized, avg_loss_R, avg_loss_t))

                    train_mae.append(mae_R.cpu().item())
                    train_mae_t.append(mae_t.cpu().item())
                    all_train_loss.append(avg_loss_R.cpu().item())
                    all_train_loss_t.append(avg_loss_t.cpu().item())
                else:
                    mae = torch.mean(torch.abs(labels - outputs))

                    epoch_avg_ec_err_truth, epoch_avg_ec_err_pred, epoch_avg_ec_err_pred_unormalized, avg_loss, epoch_penalty = (
                        v / len(train_loader) for v in (epoch_avg_ec_err_truth, epoch_avg_ec_err_pred, epoch_avg_ec_err_pred_unormalized, avg_loss, epoch_penalty))

                    train_mae.append(mae.cpu().item())
                    all_train_loss.append(avg_loss.cpu().item())
                    all_penalty.append(epoch_penalty.cpu().item())

                ec_err_truth.append(epoch_avg_ec_err_truth.cpu().item())
                ec_err_pred.append(epoch_avg_ec_err_pred.cpu().item())
                ec_err_pred_unoramlized.append(epoch_avg_ec_err_pred_unormalized.cpu().item())
                
            except Exception as e:
                print_and_write(f'7 {e}')

            # Validation
            # self.eval()
            # val_labels, val_outputs = torch.tensor([]).to(device), torch.tensor([]).to(device)
            # val_epoch_avg_ec_err_truth, val_epoch_avg_ec_err_pred, val_epoch_avg_ec_err_pred_unormalized, epoch_penalty, val_avg_loss = 0, 0, 0, 0, 0

            # with torch.no_grad():
            #     for val_first_image, val_second_image, val_label, val_unormalized_label, val_K in val_loader:
            #         try:
            #             val_first_image, val_second_image, val_label, val_unormalized_label, val_k = val_first_image.to(
            #                 device), val_second_image.to(device), val_label.to(device), val_unormalized_label.to(device), val_K.to(device)
                        
            #             unormalized_val_output, val_output, penalty = self.forward(val_first_image, val_second_image)
            #             epoch_penalty = epoch_penalty + penalty

            #             # Compute val mean epipolar constraint error
            #             val_avg_ec_err_truth, val_avg_ec_err_pred, val_avg_ec_err_pred_unormalized = get_avg_epipolar_test_errors(
            #                 val_first_image, val_second_image, val_unormalized_label, val_output, unormalized_val_output, epoch=-1, file_num=file_num)
            #             val_epoch_avg_ec_err_truth = val_epoch_avg_ec_err_truth + val_avg_ec_err_truth
            #             val_epoch_avg_ec_err_pred = val_epoch_avg_ec_err_pred + val_avg_ec_err_pred
            #             val_epoch_avg_ec_err_pred_unormalized = val_epoch_avg_ec_err_pred_unormalized + val_avg_ec_err_pred_unormalized

            #             val_avg_loss = val_avg_loss + self.L2_loss(val_output, val_label)

            #             val_outputs = torch.cat((val_outputs, val_output), dim=0)
            #             val_labels = torch.cat((val_labels, val_label), dim=0)
            #         except Exception as e:
            #             print_and_write(f'length: {len(val_labels)}, val exception: {e}')
            #     try:
            #         # Calculate and store mean absolute error for the epoch
            #         mae = torch.mean(torch.abs(val_labels - val_outputs))

            #         val_epoch_avg_ec_err_truth, val_epoch_avg_ec_err_pred_unormalized, val_epoch_avg_ec_err_pred, epoch_penalty, val_avg_loss = (
            #             v / len(val_loader) for v in (val_epoch_avg_ec_err_truth, val_epoch_avg_ec_err_pred_unormalized, val_epoch_avg_ec_err_pred, epoch_penalty, val_avg_loss))

            #         val_mae.append(mae.cpu().item())
            #         val_ec_err_truth.append(val_epoch_avg_ec_err_truth.cpu().item())
            #         val_ec_err_pred.append(val_epoch_avg_ec_err_pred.cpu().item())
            #         val_ec_err_pred_unormalized.append(val_epoch_avg_ec_err_pred_unormalized.cpu().item())
            #         all_val_loss.append(val_avg_loss.cpu().item())
            #         all_penalty.append(epoch_penalty.cpu().item())
            #     except Exception as e:
            #         print_and_write(f'8 {e}')
            try:
                if self.predict_pose:
                    epoch_output = f"""Epoch {epoch+1}/{num_epochs}, Training Loss R: {all_train_loss[-1]}, Training Loss t: {all_train_loss_t[-1]}
                    Training R MAE: {train_mae[-1]} Training t MAE: {train_mae_t[-1]}
                    Train epipolar error pred unormalized: {ec_err_pred_unoramlized[-1]}
                    Train epipolar error pred: {ec_err_pred[-1]}\n"""
                else:
                    epoch_output = f"""Epoch {epoch+1}/{num_epochs}, Training Loss: {all_train_loss[-1]} 
                    Training MAE: {train_mae[-1]}
                    Train epipolar error pred unormalized: {ec_err_pred_unoramlized[-1]}
                    Train epipolar error pred: {ec_err_pred[-1]}
                    penalty: {all_penalty[-1]}\n"""
                    # epoch_output = f"""Epoch {epoch+1}/{num_epochs}, Training Loss: {all_train_loss[-1]} Val Loss: {all_val_loss[-1]} 
                    # Training MAE: {train_mae[-1]} Val mae: {val_mae[-1]} 
                    # Train epipolar error pred unormalized: {ec_err_pred_unoramlized[-1]} Val epipolar error pred unormalized: {val_ec_err_pred_unormalized[-1]}
                    # Train epipolar error pred: {ec_err_pred[-1]} Val epipolar error pred: {val_ec_err_pred[-1]} 
                    # penalty: {all_penalty[-1]}\n"""
                print_and_write(epoch_output)
            except Exception as e:
                print_and_write(f'9 {e}')
            # If the model is not learning or outputs nan, stop training
            # if not_learning(all_train_loss, all_val_loss) or check_nan(all_train_loss[-1], all_val_loss[-1], train_mae[-1], val_mae[-1], ec_err_pred_unoramlized[-1], val_ec_err_pred_unormalized[-1], ec_err_pred[-1],all_penalty[-1]):
            #     num_epochs = epoch + 1
            #     break
        
        output = f"""Train unormalized ground truth error: {np.mean(ec_err_truth)} val unormalized ground truth error: {np.mean(val_ec_err_truth)}\n\n\n"""
        print_and_write(output)
        try:
            plot_over_epoch(x=range(1, num_epochs + 1), y1=all_train_loss, y2=all_val_loss, 
                            title="Loss" if not self.predict_pose else "Loss R", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, batchnorm_and_dropout=self.batchnorm_and_dropout, 
                            lr_mlp = self.lr_mlp, lr_vit = self.lr_vit, overfitting=self.overfitting, average_embeddings=self.average_embeddings, 
                            model=self.pretrained_model_name, augmentation=self.augmentation, enforce_rank_2=self.enforce_rank_2, predict_pose=self.predict_pose,
                            use_reconstruction=self.use_reconstruction)
            
            plot_over_epoch(x=range(1, num_epochs + 1), y1=train_mae, y2=val_mae, 
                            title="MAE" if not self.predict_pose else "MAE R", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, batchnorm_and_dropout=self.batchnorm_and_dropout, 
                            lr_mlp = self.lr_mlp, lr_vit = self.lr_vit, overfitting=self.overfitting, average_embeddings=self.average_embeddings, 
                            model=self.pretrained_model_name, augmentation=self.augmentation, enforce_rank_2=self.enforce_rank_2, predict_pose=self.predict_pose,
                            use_reconstruction=self.use_reconstruction)
            
            if self.predict_pose:
                plot_over_epoch(x=range(1, num_epochs + 1), y1=all_train_loss_t, y2=all_val_loss, 
                                title="Loss t", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, batchnorm_and_dropout=self.batchnorm_and_dropout, 
                                lr_mlp = self.lr_mlp, lr_vit = self.lr_vit, overfitting=self.overfitting, average_embeddings=self.average_embeddings, 
                                model=self.pretrained_model_name, augmentation=self.augmentation, enforce_rank_2=self.enforce_rank_2, predict_pose=self.predict_pose,
                                use_reconstruction=self.use_reconstruction)     
                
                plot_over_epoch(x=range(1, num_epochs + 1), y1=train_mae_t, y2=val_mae, 
                                title="MAE t", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, batchnorm_and_dropout=self.batchnorm_and_dropout, 
                                lr_mlp = self.lr_mlp, lr_vit = self.lr_vit, overfitting=self.overfitting, average_embeddings=self.average_embeddings, 
                                model=self.pretrained_model_name, augmentation=self.augmentation, enforce_rank_2=self.enforce_rank_2, predict_pose=self.predict_pose,
                                use_reconstruction=self.use_reconstruction)           
            
            plot_over_epoch(x=range(1, num_epochs + 1), y1=ec_err_pred_unoramlized, y2=val_ec_err_pred_unormalized, 
                            title="Epipolar error unormalized F", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, batchnorm_and_dropout=self.batchnorm_and_dropout, 
                            lr_mlp = self.lr_mlp, lr_vit = self.lr_vit, overfitting=self.overfitting, average_embeddings=self.average_embeddings, 
                            model=self.pretrained_model_name, augmentation=self.augmentation, enforce_rank_2=self.enforce_rank_2, predict_pose=self.predict_pose,
                            use_reconstruction=self.use_reconstruction)
            
            plot_over_epoch(x=range(1, num_epochs + 1), y1=ec_err_pred, y2=val_ec_err_pred, 
                            title="Epipolar error F", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, batchnorm_and_dropout=self.batchnorm_and_dropout,
                            lr_mlp = self.lr_mlp, lr_vit = self.lr_vit, overfitting=self.overfitting, average_embeddings=self.average_embeddings, 
                            model=self.pretrained_model_name, augmentation=self.augmentation, enforce_rank_2=self.enforce_rank_2, predict_pose=self.predict_pose,
                            use_reconstruction=self.use_reconstruction)
        except Exception as e:
            print_and_write(f'9 {e}')   


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
        R_x = torch.tensor([
            [1.,    0.,             0.],
            [0.,    torch.cos(x[2]),    -torch.sin(x[2])],
            [0.,    torch.sin(x[2]),     torch.cos(x[2])]
        ], requires_grad=True).to(device)

        R_y = torch.tensor([
            [torch.cos(x[3]),    0.,    -torch.sin(x[3])],
            [0.,            1.,     0.],
            [torch.sin(x[3]),    0.,     torch.cos(x[3])]
        ], requires_grad=True).to(device)

        R_z = torch.tensor([
            [torch.cos(x[4]),    -torch.sin(x[4]),    0.],
            [torch.sin(x[4]),    torch.cos(x[4]),     0.],
            [0.,            0.,             1.]
        ], requires_grad=True).to(device)

        K1_inv = torch.tensor([
                    [-1/(x[0]+1e-8),   0.,             0.],
                    [0.,            -1/(x[0]+1e-8),    0.],
                    [0.,            0.,             1.]
                ], requires_grad=True).to(device)

        K2_inv = torch.tensor([
                    [-1/(x[1]+1e-8),   0.,             0.],
                    [0.,            -1/(x[1]+1e-8),    0.],
                    [0.,            0.,             1.]
                ], requires_grad=True).to(device)

        T = torch.tensor([
                    [0.,  -x[7], x[6]],
                    [x[7],  0,   -x[5]],
                    [-x[6], x[5],  0]
                ], requires_grad=True).to(device)

        # K1_inv = self.get_inv_intrinsic(x[0])
        # K2_inv = self.get_inv_intrinsic(x[1])  # TODO: K2 should be -t not just -1..
        # R = self.get_rotation(x[2], x[3], x[4])
        # T = self.get_translate(x[5], x[6], x[7])
        R = torch.matmul(R_x, torch.matmul(R_y, R_z))
        F = torch.matmul(K2_inv,torch.matmul(R, torch.matmul(T, K1_inv)))

        return F



