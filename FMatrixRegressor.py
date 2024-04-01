from params import *
from utils import *
from FunMatrix import *
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
        
        self.L2_loss = nn.MSELoss().to(device)
        self.optimizer = optim.Adam(params, lr=lr_vit)

        if self.predict_pose:
            self.t_mlp = MLP(mlp_input_shape, mlp_hidden_sizes,
                        3, batchnorm_and_dropout).to(device)
            params_t = [
                    {'params': self.pretrained_model_t.parameters(), 'lr': lr_vit},  # Lower learning rate for the pre-trained vision transformer
                    {'params': self.t_mlp.parameters(), 'lr': lr_mlp}   # Potentially higher learning rate for the MLP
                ]  

            self.L2_loss_t = nn.MSELoss().to(device)
            self.optimizer_t = optim.Adam(params_t, lr=lr_mlp)


    def get_embeddings(self, x1, x2, predict_t=False):
        if self.clip:  
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
        else:
            x1_embeddings = self.pretrained_model(x1).last_hidden_state[:, 1:, :].view(-1,  7*7*self.model_hidden_size)
            x2_embeddings = self.pretrained_model(x2).last_hidden_state[:, 1:, :].view(-1,  7*7*self.model_hidden_size)

        if self.average_embeddings:
            try:
                avg_patches = nn.AdaptiveAvgPool2d(1)
                x1_embeddings = avg_patches(x1_embeddings.view(-1, self.model_hidden_size, 7, 7)).view(-1, self.model_hidden_size)
                x2_embeddings = avg_patches(x2_embeddings.view(-1, self.model_hidden_size, 7, 7)).view(-1, self.model_hidden_size)
            except Exception as e: 
                print_and_write(f'avg_patches: {e}')
                return

        if GROUP_CONV["use"]:
            grouped_conv_layer = GroupedConvolution(in_channels=self.model_hidden_size,   # Total input channels
                                    out_channels=GROUP_CONV["out_channels"],  # Total output channels you want
                                    kernel_size=3,
                                    padding=1,
                                    groups=GROUP_CONV["num_groups"])
            x1_embeddings = grouped_conv_layer(x1_embeddings.unsqueeze(2).unsqueeze(3)).view(-1, self.model_hidden_size//3)
            x2_embeddings = grouped_conv_layer(x2_embeddings.unsqueeze(2).unsqueeze(3)).view(-1, self.model_hidden_size//3)

        embeddings = torch.cat([x1_embeddings, x2_embeddings], dim=1)

        return embeddings

    def forward(self, x1, x2, predict_t=False):
        try:
            embeddings = self.get_embeddings(x1, x2, predict_t=predict_t)
        except Exception as e:
            print_and_write(f'get_embeddings: {e}')
            return

        # Apply MLP on embedding vectors
        try:        
            unormalized_output = self.mlp(embeddings).view(-1,3,3) if not predict_t else self.t_mlp(embeddings).view(-1,3,1)
        except Exception as e:
            print_and_write(f'mlp: {e}')
            return
        
        # Apply norm layer
        try:            
            output = norm_layer(unormalized_output.view(-1, 9)).view(-1,3,3) if not predict_t else norm_layer(unormalized_output.view(-1, 3), predict_t=True).view(-1,3,1)
        except Exception as e:
            print_and_write(f'norm_layer: {e}')
            return
        try:
            penalty = last_sing_value_penalty(unormalized_output) if not self.predict_pose else 0
        except Exception as e:
            print_and_write(f'last_sing_value_penalty: {e}')
            print_memory()
            return

        return unormalized_output, output, penalty


    def train_model(self, train_loader, val_loader, num_epochs):
        # Lists to store training statistics
        all_train_loss, all_train_loss_t, all_val_loss, train_mae, train_mae_t, val_mae, \
        all_algberaic_truth, all_algberaic_pred, all_algberaic_pred_unormalized, \
        all_RE1_truth, all_RE1_pred, all_RE1_pred_unormalized, \
        all_SED_truth, all_SED_pred, all_SED_pred_unormalized, all_penalty = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        for epoch in range(num_epochs):
            try:
                self.train()
                labels, outputs, Rs, ts = torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)
            except Exception as e:
                print_and_write(f'1 {e}')
                print_memory()
                return
            
            epoch_stats = {"algebraic_dist_truth": 0, "algebraic_dist_pred": 0, "algebraic_dist_pred_unormalized": 0, 
                            "RE1_dist_truth": 0, "RE1_dist_pred": 0, "RE1_dist_pred_unormalized": 0, 
                            "SED_dist_truth": 0, "SED_dist_pred": 0, "SED_dist_pred_unormalized": 0, 
                            "avg_loss": 0, "avg_loss_R": 0, "avg_loss_t": 0, "epoch_penalty": 0, "file_num": 0}
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
                        epoch_stats["epoch_penalty"] = epoch_stats["epoch_penalty"] + penalty

                except Exception as e:
                    print_and_write(f'3 {e}')
                    return

                try:
                    update_dists(epoch_stats, first_image.detach(), second_image.detach(), unormalized_label.detach(), output.detach(), unormalized_output.detach(), epoch)
                except Exception as e:
                    print_and_write(f'4 {e}')
                    return
                
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
                    return
            try:
                # Calculate and store mean absolute error for the epoch
                if self.predict_pose:
                    mae_R = torch.mean(torch.abs(labels[:, :, :3] - Rs))
                    mae_t = torch.mean(torch.abs(label[:, :, 3].view(-1,3,1) - ts))

                    avg_loss_R, avg_loss_t = (
                        v / len(train_loader) for v in (avg_loss_R, avg_loss_t))

                    train_mae.append(mae_R.cpu().item())
                    train_mae_t.append(mae_t.cpu().item())
                    all_train_loss.append(avg_loss_R.cpu().item())
                    all_train_loss_t.append(avg_loss_t.cpu().item())
                else:
                    mae = torch.mean(torch.abs(labels - outputs))

                    avg_loss, epoch_penalty = (
                        v / len(train_loader) for v in (avg_loss, epoch_penalty))

                    train_mae.append(mae.cpu().item())
                    all_train_loss.append(avg_loss.cpu().item())
                    all_penalty.append(epoch_penalty.cpu().item())
            
                all_algberaic_truth.append(epoch_stats["algebraic_dist_truth"].cpu().item() / len(train_loader))
                all_algberaic_pred.append(epoch_stats["algebraic_dist_pred"].cpu().item() / len(train_loader))
                all_algberaic_pred_unormalized.append(epoch_stats["algebraic_dist_pred_unormalized"].cpu().item() / len(train_loader))
                all_RE1_truth.append(epoch_stats["RE1_dist_truth"].cpu().item() / len(train_loader))
                all_RE1_pred.append(epoch_stats["RE1_dist_pred"].cpu().item() / len(train_loader))
                all_RE1_pred_unormalized.append(epoch_stats["RE1_dist_pred_unormalized"].cpu().item() / len(train_loader))
                all_SED_truth.append(epoch_stats["SED_dist_truth"].cpu().item() / len(train_loader))
                all_SED_pred.append(epoch_stats["SED_dist_pred"].cpu().item() / len(train_loader))
                all_SED_pred_unormalized.append(epoch_stats["SED_dist_pred_unormalized"].cpu().item() / len(train_loader))
                
            except Exception as e:
                print_and_write(f'7 {e}')
                return

            try:
                epoch_output = f"""Epoch {epoch+1}/{num_epochs}"""
                if self.predict_pose:
                    epoch_output += f""", Training Loss R: {all_train_loss[-1]}, Training Loss t: {all_train_loss_t[-1]}
                    Training R MAE: {train_mae[-1]} Training t MAE: {train_mae_t[-1]}\n"""
                else:
                    epoch_output += f""", Training Loss: {all_train_loss[-1]}
                    Training MAE: {train_mae[-1]} penalty: {all_penalty[-1]}\n"""
                epoch_output += f"algebraic dist truth: {all_algberaic_truth[-1]}, algebraic dist pred: {all_algberaic_pred[-1]}, algebraic dist pred unormalized: {all_algberaic_pred_unormalized[-1]},\n"
                if RE1_DIST:
                    epoch_output += F"RE1_dist_truth: {all_RE1_truth[-1]}, RE1 dist pred: {all_RE1_pred[-1]}, RE1 dist pred unormalized: {all_RE1_pred_unormalized[-1]}\n"
                if SED_DIST:
                    epoch_output += f"SED dist truth: {all_SED_truth[-1]}, SED dist pred: {all_SED_pred[-1]}, SED dist pred unormalized: {all_SED_pred_unormalized[-1]}\n"

                print_and_write(epoch_output)
            except Exception as e:
                print_and_write(f'9 {e}')

            # If the model is not learning or outputs nan, stop training
            # if not_learning(all_train_loss, all_val_loss) or check_nan(all_train_loss[-1], all_val_loss[-1], train_mae[-1], val_mae[-1], ec_err_pred_unoramlized[-1], val_ec_err_pred_unormalized[-1], ec_err_pred[-1],all_penalty[-1]):
            #     num_epochs = epoch + 1
            #     break

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
            
            plot_over_epoch(x=range(1, num_epochs + 1), y1=all_algberaic_pred_unormalized, y2=[], 
                            title="Algebraic distance unormalized F", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, batchnorm_and_dropout=self.batchnorm_and_dropout, 
                            lr_mlp = self.lr_mlp, lr_vit = self.lr_vit, overfitting=self.overfitting, average_embeddings=self.average_embeddings, 
                            model=self.pretrained_model_name, augmentation=self.augmentation, enforce_rank_2=self.enforce_rank_2, predict_pose=self.predict_pose,
                            use_reconstruction=self.use_reconstruction)
            
            plot_over_epoch(x=range(1, num_epochs + 1), y1=all_algberaic_pred, y2=[], 
                            title="Algebraic distance F", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, batchnorm_and_dropout=self.batchnorm_and_dropout,
                            lr_mlp = self.lr_mlp, lr_vit = self.lr_vit, overfitting=self.overfitting, average_embeddings=self.average_embeddings, 
                            model=self.pretrained_model_name, augmentation=self.augmentation, enforce_rank_2=self.enforce_rank_2, predict_pose=self.predict_pose,
                            use_reconstruction=self.use_reconstruction)
            if RE1_DIST:
                plot_over_epoch(x=range(1, num_epochs + 1), y1=all_RE1_pred_unormalized, y2=[], 
                                title="RE1 distance unormalized F", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, batchnorm_and_dropout=self.batchnorm_and_dropout, 
                                lr_mlp = self.lr_mlp, lr_vit = self.lr_vit, overfitting=self.overfitting, average_embeddings=self.average_embeddings, 
                                model=self.pretrained_model_name, augmentation=self.augmentation, enforce_rank_2=self.enforce_rank_2, predict_pose=self.predict_pose,
                                use_reconstruction=self.use_reconstruction)
                
                plot_over_epoch(x=range(1, num_epochs + 1), y1=all_RE1_pred, y2=[], 
                                title="RE1 distance F", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, batchnorm_and_dropout=self.batchnorm_and_dropout,
                                lr_mlp = self.lr_mlp, lr_vit = self.lr_vit, overfitting=self.overfitting, average_embeddings=self.average_embeddings, 
                                model=self.pretrained_model_name, augmentation=self.augmentation, enforce_rank_2=self.enforce_rank_2, predict_pose=self.predict_pose,
                                use_reconstruction=self.use_reconstruction)
            if SED_DIST:
                plot_over_epoch(x=range(1, num_epochs + 1), y1=all_SED_pred_unormalized, y2=[], 
                                title="SED distance unormalized F", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, batchnorm_and_dropout=self.batchnorm_and_dropout, 
                                lr_mlp = self.lr_mlp, lr_vit = self.lr_vit, overfitting=self.overfitting, average_embeddings=self.average_embeddings, 
                                model=self.pretrained_model_name, augmentation=self.augmentation, enforce_rank_2=self.enforce_rank_2, predict_pose=self.predict_pose,
                                use_reconstruction=self.use_reconstruction)
                
                plot_over_epoch(x=range(1, num_epochs + 1), y1=all_SED_pred, y2=[], 
                                title="SED distance F", penalty_coeff=self.penalty_coeff, batch_size=self.batch_size, batchnorm_and_dropout=self.batchnorm_and_dropout,
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