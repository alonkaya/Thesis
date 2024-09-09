from params import *
from utils import *
import torch.optim as optim
from transformers import CLIPVisionModel, CLIPVisionConfig, ResNetModel


class AffineRegressor(nn.Module):
    def __init__(self, lr, batch_size, alpha, avg_embeddings=AVG_EMBEDDINGS, model_name=MODEL, plots_path=None, pretrained_path=PRETRAINED_PATH, use_conv=USE_CONV, num_epochs=NUM_EPOCHS):

        """
        Args:
        - lr: learning rate for the vision transformer
        - lr_mlp: learning rate for the MLP
        - batch_size: batch size for training
        - deepF_noCorrs: whether to use the deepF_noCorrs model
        - augmentation: whether to use data augmentation
        - model_name: name of the model to use
        - unfrozen_layers: number of layers to unfreeze
        - use_reconstruction: whether to use the reconstruction layer
        - pretrained_path: path to a pretrained model
        - alg_coeff: coefficient for the algebraic distance
        - re1_coeff: coefficient for the RE1 distance
        - sed_coeff: coefficient for the SED distance
        - plots_path: path to save the plots
        - use_conv: whether to use a convolutional layer
        - num_epochs: number of epochs to train
        """

        super(AffineRegressor, self).__init__()
        self.to(device)
        self.batch_size = batch_size
        self.lr = lr
        self.model_name = model_name
        self.alpha = alpha
        self.plots_path = plots_path
        self.use_conv = use_conv
        self.num_epochs = num_epochs
        self.avg_embeddings = avg_embeddings
        self.start_epoch = 0

        # Lists to store training statistics
        self.all_train_loss, self.all_val_loss, \
        self.all_train_mae_shift, self.all_val_mae_shift,\
        self.all_train_euclidean_shift, self.all_val_euclidean_shift, \
        self.all_train_mae_angle, self.all_val_mae_angle, \
        self.all_train_mse_angle, self.all_val_mse_angle = [], [], [], [], [], [], [], [], [], []
        
        self.resnet = False
        if model_name == CLIP_MODEL_NAME:
            # Initialize CLIP processor and pretrained model
            if TRAIN_FROM_SCRATCH:
                config = CLIPVisionConfig()
                self.model = CLIPVisionModel(config).to(device)
            else:
                self.model = CLIPVisionModel.from_pretrained(model_name).to(device)

        elif model_name == RESNET_MODEL_NAME:
            self.resnet = True
            self.model = ResNetModel.from_pretrained(model_name).to(device)
    

        if pretrained_path or os.path.exists(os.path.join(plots_path, 'model.pth')): 
            model_path = os.path.join(pretrained_path, 'model.pth') if pretrained_path else os.path.join(plots_path, 'model.pth')
            self.load_model(model_path=model_path)

        else:
            # Get input dimension for the MLP based on ViT configuration
            self.hidden_size = self.model.config.hidden_size if not self.resnet else self.model.config.hidden_sizes[-1]
            self.num_patches = self.model.config.image_size // self.model.config.patch_size if not self.resnet else 7
            mlp_input_shape = 2 * (self.num_patches**2) * self.hidden_size 

            # Initialize loss functions
            self.L2_loss = nn.MSELoss().to(device)
            self.huber_loss = nn.HuberLoss().to(device)
            
            if self.avg_embeddings:
                mlp_input_shape //= (self.num_patches**2) 

            elif self.use_conv:
                self.conv = ConvNet(input_dim= 2*self.hidden_size, batch_size=self.batch_size).to(device)
                mlp_input_shape = 2 * self.conv.hidden_dims[-1] * 3 * 3 

            # Initialize MLP
            self.mlp = MLP(input_dim=mlp_input_shape).to(device)

            # Load optimizer and scheduler
            self.optimizer = optim.Adam([
                {'params': self.model.parameters(), 'lr': self.lr},
                {'params': self.mlp.parameters(), 'lr': self.lr},   
                {'params': self.conv.parameters(), 'lr': self.lr} if self.use_conv else {'params': []}
            ])

        self.to(device)

    def FeatureExtractor(self, x1, x2):
        if torch.isnan(x1).any():
            print_and_write(f"0. Nan in x1_embeddings\n {x1}")

        # Run ViT. Input shape x1,x2 are (batch_size, channels, height, width)
        x1_embeddings = self.model(pixel_values=x1).last_hidden_state
        x2_embeddings = self.model(pixel_values=x2).last_hidden_state

        if not self.resnet:
            x1_embeddings = x1_embeddings[:, 1:, :] # Eliminate the CLS token for ViTs
            x2_embeddings = x2_embeddings[:, 1:, :] # Eliminate the CLS token for ViTs

        x1_embeddings = x1_embeddings.reshape(-1, self.hidden_size * self.num_patches * self.num_patches)
        x2_embeddings = x2_embeddings.reshape(-1, self.hidden_size * self.num_patches * self.num_patches)
        if torch.isnan(x1_embeddings).any():
            print_and_write("1. Nan in x1_embeddings", self.plots_path)
        
        if self.avg_embeddings:
            # Input shape is (batch_size, self.hidden_size, self.num_patches, self.num_patches). Output shape is (batch_size, self.hidden_size)
            avg_patches = nn.AdaptiveAvgPool2d(1)
            x1_embeddings = avg_patches(x1_embeddings.reshape(-1, self.hidden_size, self.num_patches, self.num_patches)).reshape(-1, self.hidden_size)
            x2_embeddings = avg_patches(x2_embeddings.reshape(-1, self.hidden_size, self.num_patches, self.num_patches)).reshape(-1, self.hidden_size)
            if torch.isnan(x1_embeddings).any():
                print_and_write("2. Nan in x1_embeddings", self.plots_path)

        if self.use_conv:
            # Input shape is (batch_size, self.hidden_size * 2, self.num_patches, self.num_patches). Output shape is (batch_size, 2 * CONV_HIDDEN_DIM[-1] * 3 * 3)
            x1_embeddings = x1_embeddings.reshape(-1, self.hidden_size, self.num_patches, self.num_patches)
            x2_embeddings = x2_embeddings.reshape(-1, self.hidden_size, self.num_patches, self.num_patches)
            embeddings = torch.cat([x1_embeddings, x2_embeddings], dim=1)
            embeddings = self.conv(embeddings)
        else:
            embeddings = torch.cat([x1_embeddings, x2_embeddings], dim=1)
        
        return embeddings

    def forward(self, x1, x2):
        # x1, x2 shape is (batch_size, channels, height, width)
        embeddings = self.FeatureExtractor(x1, x2) # Output shape is (batch_size, -1)
        # output shape is (batch_size, 3)
        output = self.mlp(embeddings)
        if torch.isnan(output).any():
            print_and_write(f"3. Nan in output\n{embeddings}", self.plots_path)

        output = norm_layer(output)
        if torch.isnan(output).any():
            print_and_write("4. Nan in output", self.plots_path)

        return output

    def train_model(self, train_loader, val_loader, test_loader):
        for epoch in range(self.start_epoch, self.num_epochs):
            epoch_stats = {"mae_shift": torch.tensor(0, dtype=torch.float32), "euclidean_shift": torch.tensor(0, dtype=torch.float32), \
                           "loss": torch.tensor(0, dtype=torch.float32), "mae_angle": torch.tensor(0, dtype=torch.float32), "mse_angle": torch.tensor(0, dtype=torch.float32),
                           "val_mae_shift": torch.tensor(0, dtype=torch.float32), "val_euclidean_shift": torch.tensor(0, dtype=torch.float32), \
                           "val_loss": torch.tensor(0, dtype=torch.float32), "val_mae_angle": torch.tensor(0, dtype=torch.float32), "val_mse_angle": torch.tensor(0, dtype=torch.float32)}
            send_to_device(epoch_stats)

            # Training
            self.train()
            self.dataloader_step(train_loader, epoch, epoch_stats, data_type="train")

            # Validation
            self.eval()
            with torch.no_grad():
                self.dataloader_step(val_loader, epoch, epoch_stats, data_type="val")

            # Divide by the number of batches
            divide_by_dataloader(epoch_stats, len(train_loader), len(val_loader))

            # Append epoch statistics to lists 
            self.append_epoch_stats(epoch_stats)

            print_and_write(f"""Epoch {epoch+1}/{self.num_epochs}: Training Loss: {self.all_train_loss[-1]}\t\t Val Loss: {self.all_val_loss[-1]}\n""", self.plots_path)

            if NUM_OUTPUT == 1 or NUM_OUTPUT == 3:
                print_and_write(f"""Training MAE Angle: {self.all_train_mae_angle[-1]}\t\t Val MAE Angle: {self.all_val_mae_angle[-1]}
Training MSE Angle: {self.all_train_mse_angle[-1]}\t\t Val MSE Angle: {self.all_val_mse_angle[-1]}\n""", self.plots_path)
            elif NUM_OUTPUT == 2 or NUM_OUTPUT == 3:
                print_and_write(f"""Training MAE Shift: {self.all_train_mae_shift[-1]}\t\t Val MAE Shift: {self.all_val_mae_shift[-1]}
Training Euclidean Shift: {self.all_train_euclidean_shift[-1]}\t\t Val Euclidean Shift: {self.all_val_euclidean_shift[-1]}\n""", self.plots_path)
            print_and_write("\n\n", self.plots_path)

            if check_nan(self.all_train_loss[-1], self.all_val_loss[-1], self.plots_path):
                self.num_epochs = epoch + 1
                break

            if SAVE_MODEL:
                self.save_model(epoch+1)
        
        self.test(test_loader)

        plot(x=range(1, self.num_epochs + 1), y1=self.all_train_loss, y2=self.all_val_loss, title="Loss", plots_path=self.plots_path)
        plot(x=range(1, self.num_epochs + 1), y1=self.all_train_mae_shift, y2=self.all_val_mae_shift, title="MAE Shift", plots_path=self.plots_path)
        plot(x=range(1, self.num_epochs + 1), y1=self.all_train_euclidean_shift, y2=self.all_val_euclidean_shift, title="Euclidean Shift", plots_path=self.plots_path)
        plot(x=range(1, self.num_epochs + 1), y1=self.all_train_mae_angle, y2=self.all_val_mae_angle, title="MAE Angle", plots_path=self.plots_path)
        plot(x=range(1, self.num_epochs + 1), y1=self.all_train_mse_angle, y2=self.all_val_mse_angle, title="MSE Angle", plots_path=self.plots_path)
           
    def save_model(self, epoch):
        checkpoint_path = os.path.join(self.plots_path, "model.pth")
        torch.save({
            'mlp': self.mlp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'vit': self.model.state_dict() ,
            'conv': self.conv.state_dict() if self.use_conv else '',
            'alpha': self.alpha,
            'avg_embeddings': self.avg_embeddings,
            "batch_size" : self.batch_size,
            "lr" : self.lr,
            "model_name" : self.model_name,
            "plots_path" : self.plots_path,
            "use_conv" : self.use_conv,
            "hidden_size" : self.hidden_size,
            "num_patches" : self.num_patches,
            'epoch' : epoch,
            'angle_range' : ANGLE_RANGE,
            'shift_range' : SHIFT_RANGE,
            "all_train_loss" : self.all_train_loss, 
            "all_val_loss" : self.all_val_loss,
            "all_train_mae_shift" : self.all_train_mae_shift,
            "all_val_mae_shift" : self.all_val_mae_shift,
            "all_train_euclidean_shift" : self.all_train_euclidean_shift,
            "all_val_euclidean_shift" : self.all_val_euclidean_shift,
            "all_train_mae_angle" : self.all_train_mae_angle,
            "all_val_mae_angle" : self.all_val_mae_angle,
            "all_train_mse_angle" : self.all_train_mse_angle,
            "all_val_mse_angle" : self.all_val_mse_angle,
        }, checkpoint_path) 

    def load_model(self, model_path=None):
        checkpoint = torch.load(model_path, map_location='cpu')

        self.batch_size = checkpoint.get("batch_size", self.batch_size)
        self.lr = checkpoint.get("lr", self.lr)
        self.model_name = checkpoint.get("model_name", self.model_name)
        self.plots_path = checkpoint.get("plots_path", None) if GET_OLD_PATH else self.plots_path 
        self.use_conv = checkpoint.get("use_conv", self.use_conv)
        self.avg_embeddings = checkpoint.get("avg_embeddings", self.avg_embeddings)
        self.hidden_size = checkpoint.get("hidden_size", 0)
        self.num_patches = checkpoint.get("num_patches", 0)
        self.start_epoch = checkpoint.get("epoch", 0)
        self.alpha = checkpoint.get("alpha", 0)
        self.all_train_loss = checkpoint.get("all_train_loss", [])
        self.all_val_loss = checkpoint.get("all_val_loss", [])
        self.all_train_mae_shift = checkpoint.get("all_train_mae_shift", [])
        self.all_val_mae_shift = checkpoint.get("all_val_mae_shift", [])
        self.all_train_euclidean_shift = checkpoint.get("all_train_euclidean_shift", [])
        self.all_val_euclidean_shift = checkpoint.get("all_val_euclidean_shift", [])
        self.all_train_mae_angle = checkpoint.get("all_train_mae_angle", [])
        self.all_val_mae_angle = checkpoint.get("all_val_mae_angle", [])
        self.all_train_mse_angle = checkpoint.get("all_train_mse_angle", [])
        self.all_val_mse_angle = checkpoint.get("all_val_mse_angle", [])

        # Get input dimension for the MLP based on ViT configuration
        self.hidden_size = self.model.config.hidden_size if not self.resnet else self.model.config.hidden_sizes[-1]
        self.num_patches = self.model.config.image_size // self.model.config.patch_size if not self.resnet else 7
        mlp_input_shape = 2 * (self.num_patches**2) * self.hidden_size 

        # Initialize loss functions
        self.L2_loss = nn.MSELoss().to(device)
        self.huber_loss = nn.HuberLoss().to(device)

        # Load conv/average embeddings
        if self.use_conv:
            self.conv = ConvNet(input_dim= 2*self.hidden_size, batch_size=self.batch_size).to(device)
            mlp_input_shape = 2 * self.conv.hidden_dims[-1] * 3 * 3 
            self.conv.load_state_dict(checkpoint['conv'])
            self.conv.to(device)
        
        if self.avg_embeddings:
            mlp_input_shape //= (self.num_patches**2) 

        # Load MLP
        self.mlp = MLP(input_dim=mlp_input_shape).to(device)
        self.mlp.load_state_dict(checkpoint['mlp'])
        self.mlp.to(device)

        # Load model
        self.model.load_state_dict(checkpoint['vit']) 
        self.model.to(device)

        # Load optimizer and scheduler
        self.optimizer = optim.Adam([
            {'params': self.model.parameters(), 'lr': self.lr},
            {'params': self.mlp.parameters(), 'lr': self.lr},   # Potentially higher learning rate for the MLP
            {'params': self.conv.parameters(), 'lr': self.lr} if self.use_conv else {'params': []}
        ])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def dataloader_step(self, dataloader, epoch, epoch_stats, data_type):
        prefix = "val_" if data_type == "val" else "test_" if data_type == "test" else ""
        for batch in dataloader:
            if NUM_OUTPUT == 1: # Angle
                img1, img2, angle = batch
                img1, img2, angle = img1.to(device), img2.to(device), angle.to(device)
            elif  NUM_OUTPUT == 2: # Shift
                img1, img2, shift_x, shift_y = batch
                img1, img2, shift = img1.to(device), img2.to(device), torch.stack([shift_x, shift_y], dim=1).to(device)
            elif NUM_OUTPUT == 3: # Angle and Shift
                img1, img2, angle, shift_x, shift_y = batch
                img1, img2, angle, shift = img1.to(device), img2.to(device), angle.to(device), torch.stack([shift_x, shift_y], dim=1).to(device)

            # Forward pass
            output = self.forward(img1, img2)

            # Compute loss
            if NUM_OUTPUT == 1 or NUM_OUTPUT == 3:
                mse_angle = self.L2_loss(output[:,0], angle)
                loss = mse_angle
            elif NUM_OUTPUT == 2 or NUM_OUTPUT == 3:
                mse_shift = self.L2_loss(output[:, 1:], shift)
                loss = mse_shift
            if NUM_OUTPUT == 3:
                loss = mse_angle + self.alpha * mse_shift

            with torch.no_grad():
                mae_angle = 0 if NUM_OUTPUT==2 else torch.mean(torch.abs(output[:,0] - angle))
                mae_shift = 0 if NUM_OUTPUT==1 else torch.mean(torch.abs(output[:,1:] - shift))
                euclidean_shift = 0 if NUM_OUTPUT==1 else torch.mean(torch.sqrt(torch.sum((output[:,1:] - shift)**2, dim=1)))


            if data_type == "train":
                # Compute Backward pass and gradients
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Extend lists with batch statistics
            epoch_stats[f'{prefix}mae_shift'] += mae_shift
            epoch_stats[f'{prefix}euclidean_shift'] += euclidean_shift
            epoch_stats[f'{prefix}mae_angle'] += mae_angle
            epoch_stats[f'{prefix}mse_angle'] += mse_angle
            epoch_stats[f'{prefix}loss'] += loss
        

    def append_epoch_stats(self, epoch_stats):
        self.all_train_loss.append(epoch_stats["loss"])
        self.all_val_loss.append(epoch_stats["val_loss"])

        if NUM_OUTPUT == 1 or NUM_OUTPUT == 3:
            self.all_train_mae_angle.append(epoch_stats["mae_angle"])
            self.all_val_mae_angle.append(epoch_stats["val_mae_angle"])
            self.all_train_mse_angle.append(epoch_stats["mse_angle"])
            self.all_val_mse_angle.append(epoch_stats["val_mse_angle"])

        if NUM_OUTPUT == 2 or NUM_OUTPUT == 3:
            self.all_train_mae_shift.append(epoch_stats["mae_shift"])
            self.all_val_mae_shift.append(epoch_stats["val_mae_shift"])
            self.all_train_euclidean_shift.append(epoch_stats["euclidean_shift"])
            self.all_val_euclidean_shift.append(epoch_stats["val_euclidean_shift"])


    def test(self, test_loader):
        with torch.no_grad():
            loss, mae_shift, euclidean_shift, mae_angle, mse_angle = torch.tensor(0, dtype=torch.float32), torch.tensor(0, dtype=torch.float32), torch.tensor(0, dtype=torch.float32), torch.tensor(0, dtype=torch.float32), torch.tensor(0, dtype=torch.float32)
            for epoch in range(10):
                epoch_stats = {"test_loss": torch.tensor(0, dtype=torch.float32), "test_mae_shift": torch.tensor(0, dtype=torch.float32), "test_euclidean_shift": torch.tensor(0, dtype=torch.float32), \
                               "test_mae_angle": torch.tensor(0, dtype=torch.float32), "test_mse_angle": torch.tensor(0, dtype=torch.float32)}
                send_to_device(epoch_stats)
    
                self.dataloader_step(test_loader, 0, epoch_stats, data_type="test")

                divide_by_dataloader(epoch_stats, len_test_loader=len(test_loader))

                loss += epoch_stats["test_loss"]
                mae_shift += epoch_stats["test_mae_shift"]
                euclidean_shift += epoch_stats["test_euclidean_shift"]
                mae_angle += epoch_stats["test_mae_angle"]
                mse_angle += epoch_stats["test_mse_angle"]


        print_and_write(f"""## TEST RESULTS: ##
Test Loss: {loss/10} 
Test MAE Shift: {mae_shift/10}\t\t Test Euclidean Shift: {euclidean_shift/10}
Test MAE Angle: {mae_angle/10}\t\t Test MSE Angle: {mse_angle/10}\n\n""", self.plots_path)
    

