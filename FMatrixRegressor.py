import shutil
from params import *
from utils import *
from FunMatrix import *
import torch.optim as optim
from torchvision import models
from transformers import ViTModel, CLIPVisionModel, CLIPVisionConfig, ResNetModel, AutoModel

class FMatrixRegressor(nn.Module):
    def __init__(self, lr, batch_size, L2_coeff, huber_coeff, min_lr=MIN_LR, average_embeddings=AVG_EMBEDDINGS, 
                 augmentation=AUGMENTATION, model_name=MODEL, trained_vit=TRAINED_VIT, kitti2sceneflow=KITTI2SCENEFLOW, sceneflow2kitti=SCENEFLOW2KITTI,
                 frozen_layers=0, frozen_high_layers=0, use_reconstruction=USE_RECONSTRUCTION_LAYER, cc=CC, pretrained_path=None, 
                 alg_coeff=0, re1_coeff=0, sed_coeff=0, plots_path=None, use_conv=USE_CONV, num_epochs=0):

        """
        Args:
        - lr: learning rate for the vision transformer
        - lr_mlp: learning rate for the MLP
        - average_embeddings: whether to average the embeddings of the patches
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

        super(FMatrixRegressor, self).__init__()
        self.to(device)
        self.batch_size = batch_size
        self.lr = lr
        self.min_lr = min_lr
        self.average_embeddings = average_embeddings
        self.model_name = model_name
        self.augmentation = augmentation
        self.use_reconstruction=use_reconstruction
        self.re1_coeff = re1_coeff
        self.alg_coeff = alg_coeff
        self.sed_coeff = sed_coeff
        self.L2_coeff = L2_coeff
        self.huber_coeff = huber_coeff
        self.plots_path = pretrained_path if pretrained_path else plots_path 
        self.use_conv = use_conv
        self.num_epochs = num_epochs
        self.frozen_layers = frozen_layers
        self.frozen_high_layers = frozen_high_layers
        self.start_epoch = 0
        self.cc = cc
        self.trained_vit = trained_vit # This is for when wanting to fine-tune an already trained vit 
                                       #(for example fine-tuning a vit which had been trained on the affine transfomration task)

        # Lists to store training statistics
        self.all_train_loss, self.all_val_loss, \
        self.all_train_mae, self.all_val_mae, \
        self.all_algebraic_pred, self.all_val_algebraic_pred, \
        self.all_RE1_pred, self.all_val_RE1_pred, \
        self.all_SED_pred, self.all_val_SED_pred = [], [], [], [], [], [], [], [], [], []
        
        self.resnet = False
        if model_name == CLIP_MODEL_NAME or model_name == CLIP_MODEL_NAME_16:
            # Initialize CLIP processor and pretrained model
            if TRAIN_FROM_SCRATCH:
                config = CLIPVisionConfig()
                self.model = CLIPVisionModel(config).to(device)
            else:
                self.model = CLIPVisionModel.from_pretrained(model_name).to(device)

        elif model_name == RESNET_MODEL_NAME:
            self.resnet = True
            self.model = ResNetModel.from_pretrained(model_name).to(device)
        elif model_name == DINO:
            self.model = AutoModel.from_pretrained(model_name).to(device)
        elif model_name == EFFICIENTNET:
            weights = models.efficientnet.EfficientNet_V2_M_Weights.DEFAULT
            self.model = models.efficientnet_v2_m(weights=weights).to(device)
        else:
            # Initialize ViT pretrained model
            self.model = ViTModel.from_pretrained(model_name).to(device)
            
        # Freeze frozen_layers layers
        if MODEL==CLIP_MODEL_NAME:
            for layer_idx, layer in enumerate(self.model.vision_model.encoder.layers):
                if layer_idx < self.frozen_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
                elif layer_idx >= len(self.model.vision_model.encoder.layers) - self.frozen_high_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        ## THIS IS ONLY FOR CONTINUING TRAINING FROM A EARLY STOPPED CHECKPOINT!
        self.parent_model_path = os.path.join("/mnt/sda2/Alon", self.plots_path) if COMPUTER==0 else os.path.join("/mnt_hdd15tb/alonkay/Thesis", self.plots_path) if COMPUTER==1 else self.plots_path
        if pretrained_path or os.path.exists(os.path.join(self.parent_model_path, 'model.pth')): 
            self.load_model(self.parent_model_path)
        
        elif kitti2sceneflow or sceneflow2kitti:
                p = FLYING_MODEL_PATH if sceneflow2kitti else KITTI_MODEL_PATH
                kitti_path = os.path.join("/mnt/sda2/Alon", p) if COMPUTER==0 else os.path.join("/mnt_hdd15tb/alonkay/Thesis", p) if COMPUTER==1 else p
                self.load_model(kitti_path, continue_training=False)
                for param in self.model.parameters():
                    param.requires_grad = False

        else:
            if self.trained_vit != None:
                # This is for when wanting to fine-tune an already trained vit 
                # for example fine-tuning a vit which had been trained on the affine transfomration task, on the FMatrix task
                self.trained_vit = os.path.join("/mnt/sda2/Alon", self.trained_vit) if COMPUTER==0 else self.trained_vit
                checkpoint = torch.load(self.trained_vit, map_location='cpu')
                self.model.load_state_dict(checkpoint['vit']) 
                self.model.to(device)

            # Initialize loss functions
            self.L2_loss = nn.MSELoss().to(device)
            self.huber_loss = nn.HuberLoss().to(device)

            # Get input dimension for the MLP based on ViT configuration
            self.hidden_size = self.model.config.hidden_sizes[-1] if self.resnet else 1280 if model_name==EFFICIENTNET else self.model.config.hidden_size
            self.num_patches = 7 if self.resnet or model_name==EFFICIENTNET else self.model.config.image_size // self.model.config.patch_size   

            if self.use_conv:
                convnet_input_dim = 1 if self.cc else 2*self.hidden_size 
                self.conv = ConvNet(input_dim=convnet_input_dim, batch_size=self.batch_size).to(device)
                mlp_input_shape = 2 * self.conv.hidden_dims[-1] * MAX_POOL_SIZE**2 

            # Initialize MLP
            self.mlp = MLP(input_dim=mlp_input_shape).to(device)

            # Load optimizer and scheduler
            self.optimizer = optim.Adam([
                {'params': self.model.parameters(), 'lr': self.lr},
                {'params': self.mlp.parameters(), 'lr': self.lr},   
                {'params': self.conv.parameters(), 'lr': self.lr} if self.use_conv else {'params': []}
            ])
            self.scheduler = None

        self.to(device)

    def FeatureExtractor(self, x1, x2):
        # Run ViT. Input shape x1,x2 are (batch_size, channels, height, width)
        x1_embeddings = self.model.features(x1) if self.model_name==EFFICIENTNET else self.model(x1).last_hidden_state
        x2_embeddings = self.model.features(x2) if self.model_name==EFFICIENTNET else self.model(x2).last_hidden_state

        if not self.resnet and not self.model_name==EFFICIENTNET:
            x1_embeddings = x1_embeddings[:, 1:, :] # Eliminate the CLS token for ViTs
            x2_embeddings = x2_embeddings[:, 1:, :] # Eliminate the CLS token for ViTs

        if self.cc:
            x1_embeddings = x1_embeddings.reshape(-1, 1, self.num_patches**2, self.hidden_size) # [batch, 1, num_patches^2, hidden_size]
            x2_embeddings = x2_embeddings.reshape(-1, self.num_patches**2, self.hidden_size, 1) # [batch, num_patches^2, hidden_size, 1]

            # Multipling each vector in x1 with the whole matrix x2 results in Shape: [batch, num_patches^2, num_patches^2, 1]
            cc = torch.matmul(x1_embeddings, x2_embeddings).reshape(-1, 1, self.num_patches**2, self.num_patches**2)  # [batch, num_patches^2, num_patches^2, 1]

            # Input shape is (batch_size, 1, self.num_patches^2, self.num_patches^2). Output shape is (batch_size, 2 * CONV_HIDDEN_DIM[-1] * 3 * 3)
            embeddings = self.conv(cc)

        else:
            # Input shape is (batch_size, self.hidden_size*2, self.num_patches, self.num_patches). Output shape is (batch_size, 2 * CONV_HIDDEN_DIM[-1] * 3 * 3)
            x1_embeddings = x1_embeddings.reshape(-1, self.hidden_size, self.num_patches, self.num_patches)
            x2_embeddings = x2_embeddings.reshape(-1, self.hidden_size, self.num_patches, self.num_patches)
            embeddings = torch.cat([x1_embeddings, x2_embeddings], dim=1)
            embeddings = self.conv(embeddings)
        
        return embeddings

    def forward(self, x1, x2):
        # x1, x2 shape is (batch_size, channels, height, width)
        embeddings = self.FeatureExtractor(x1, x2) # Output shape is (batch_size, -1)

        output = self.mlp(embeddings)

        output = paramterization_layer(output.view(-1,8), self.plots_path) if self.use_reconstruction else output

        output = norm_layer(output.view(-1, 9)).view(-1,3,3) 

        return output


    def train_model(self, train_loader, val_loader, test_loader):
        break_when_good = False
        for epoch in range(self.start_epoch, self.num_epochs**2):
            epoch_stats = {"algebraic_pred": torch.tensor(0), "algebraic_sqr_pred": torch.tensor(0), "RE1_pred": torch.tensor(0), "SED_pred": torch.tensor(0), 
                            "val_algebraic_pred": torch.tensor(0), "val_algebraic_sqr_pred": torch.tensor(0), "val_RE1_pred": torch.tensor(0), "val_SED_pred": torch.tensor(0), 
                            "algebraic_truth": torch.tensor(0), "algebraic_sqr_truth": torch.tensor(0), "RE1_truth": torch.tensor(0), "SED_truth": torch.tensor(0), 
                            "val_algebraic_truth": torch.tensor(0), "val_algebraic_sqr_truth": torch.tensor(0), "val_RE1_truth": torch.tensor(0), "val_SED_truth": torch.tensor(0), 
                            "loss": torch.tensor(0), "val_loss": torch.tensor(0),
                            "labels": torch.tensor([]), "outputs": torch.tensor([]), "val_labels": torch.tensor([]), "val_outputs": torch.tensor([]),
                            "file_num": 0}
            send_to_device(epoch_stats)

            # Training
            self.train()
            self.dataloader_step(train_loader, epoch, epoch_stats, data_type="train")

            # Validation
            self.eval()
            with torch.no_grad():
                self.dataloader_step(val_loader, epoch, epoch_stats, data_type="val")
                    
            train_mae = torch.mean(torch.abs(epoch_stats["labels"] - epoch_stats["outputs"]))
            val_mae = torch.mean(torch.abs(epoch_stats["val_labels"] - epoch_stats["val_outputs"]))

            divide_by_dataloader(epoch_stats, len(train_loader), len(val_loader))

            self.append_epoch_stats(train_mae.cpu().item(), val_mae.cpu().item(), epoch_stats)

            if (self.optimizer.param_groups[0]['lr'] > self.min_lr or SCHED == "cosine") and self.scheduler != None:
                self.scheduler.step()

            if epoch == 0: 
                print_and_write(f"""algebraic_truth: {epoch_stats["algebraic_truth"]}\t\t val_algebraic_truth: {epoch_stats["val_algebraic_truth"]}
RE1_truth: {epoch_stats["RE1_truth"]}\t\t val_RE1_truth: {epoch_stats["val_RE1_truth"]}
SED_truth: {epoch_stats["SED_truth"]}\t\t val_SED_truth: {epoch_stats["val_SED_truth"]}\n""", self.plots_path)

            print_and_write(f"""Epoch {epoch+1}/{self.num_epochs}: Training Loss: {self.all_train_loss[-1]}\t\t Val Loss: {self.all_val_loss[-1]}
             \tTraining MAE: {self.all_train_mae[-1]}\t\t Val MAE: {self.all_val_mae[-1]}
             \tAlgebraic dist: {self.all_algebraic_pred[-1]}\t\t Val Algebraic dist: {self.all_val_algebraic_pred[-1]}
             \tRE1 dist: {self.all_RE1_pred[-1]}\t\t Val RE1 dist: {self.all_val_RE1_pred[-1]}
             \tSED dist: {self.all_SED_pred[-1]}\t\t Val SED dist: {self.all_val_SED_pred[-1]}\n""", self.plots_path)

                
            # Found Nan 
            if check_nan(self.all_train_loss[-1], self.all_val_loss[-1], self.all_train_mae[-1], self.all_val_mae[-1], self.plots_path):
                self.num_epochs = epoch + 1
                break
            
            # BAD PLOTS
            if STEREO and epoch > int(self.num_epochs * 2/5) and not_learning(self.all_val_RE1_pred, self.plots_path):
                self.num_epochs = epoch + 1
                os.rename(self.plots_path, self.plots_path + "__bad")
                self.plots_path = self.plots_path + "__bad"
                if COMPUTER==0 or COMPUTER==1:
                    try:
                        os.rename(self.parent_model_path, self.parent_model_path + "__bad")
                    except Exception as e:
                        print_and_write(f"Renaming failed: {e}", self.plots_path)
                        break
                self.parent_model_path = self.parent_model_path + "__bad"
                print_and_write("\nModel not learning and is very bad, stopping training\n", self.plots_path)
                break

            if SAVE_MODEL: ## This saves the model 100 times in total 
                self.save_model(epoch+1)
            
            # If the last epochs are not decreasing in val loss, raise break_when_good flag
            if (self.resnet and epoch > int(self.num_epochs * 3/5) and not_decreasing(self.all_val_loss, self.num_epochs, self.plots_path)) or \
                (not self.resnet and epoch > int(self.num_epochs * 3/4) and not_decreasing(self.all_val_loss, self.num_epochs, self.plots_path)) or \
                epoch > self.num_epochs:
                break_when_good = True

            # If last epoch got best results of psat 4 epochs, stop training
            if break_when_good and ready_to_break(self.all_val_loss, self.num_epochs):
                self.num_epochs = epoch + 1
                break
        
        self.save_model(epoch+1, definetly=True) 
        self.test(test_loader)

        try:
            self.plot_all()
        except Exception as e:
            print_and_write(f"Plotting failed: {e}", self.plots_path)
        
    def dataloader_step(self, dataloader, epoch, epoch_stats, data_type):
        prefix = "val_" if data_type == "val" else "test_" if data_type == "test" else ""
        batch_alg_preds, batch_SED_preds, batch_RE1_preds = [], [], []
        for img1, img2, label, pts1, pts2, _,  in dataloader:
            img1, img2, label, pts1, pts2 = img1.to(device), img2.to(device), label.to(device), pts1.to(device), pts2.to(device)

            # Forward pass
            output = self.forward(img1, img2)

            if data_type == "train":
                pts1.requires_grad = True
                pts2.requires_grad = True
            # Update epoch statistics
            batch_alg_pred, batch_SED_pred, batch_RE1_pred  = update_epoch_stats(
                epoch_stats, img1.detach(), img2.detach(), label.detach(), output, pts1, pts2, data_type, epoch)
            
            if data_type=="test":
                batch_alg_preds.append(batch_alg_pred.detach())
                batch_SED_preds.append(batch_SED_pred.detach())
                batch_RE1_preds.append(batch_RE1_pred.detach())
            
            # Compute loss
            loss = self.L2_coeff*self.L2_loss(output, label) + self.huber_coeff*self.huber_loss(output, label) + \
                    self.sed_coeff*batch_SED_pred
            epoch_stats[f'{prefix}loss'] = epoch_stats[f'{prefix}loss'] + loss.detach()

            if data_type == "train":
                # Compute Backward pass and gradients
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Extend lists with batch statistics
            epoch_stats[f'{prefix}labels'] = torch.cat((epoch_stats[f'{prefix}labels'], label.detach()), dim=0)
            epoch_stats[f'{prefix}outputs'] = torch.cat((epoch_stats[f'{prefix}outputs'], output.detach()), dim=0)

        return batch_alg_preds, batch_SED_preds, batch_RE1_preds

    def save_model(self, epoch, definetly=False):
        model_path = os.path.join(self.parent_model_path, "model.pth")
        os.makedirs(self.parent_model_path, exist_ok=True)
        # Backup previous checkpoint
        if os.path.exists(model_path) and epoch % (self.num_epochs//10) == 0: 
            backup_path = os.path.join(self.parent_model_path, "backup_model.pth")
            shutil.copy(model_path, backup_path)
        if definetly or epoch % (self.num_epochs//100) == 0:
            torch.save({
                'mlp': self.mlp.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'vit': self.model.state_dict() ,
                'conv': self.conv.state_dict() if self.use_conv else '',
                "scheduler" : None if self.scheduler==None else self.scheduler.state_dict(),
                "L2_coeff" : self.L2_coeff,
                "huber_coeff" : self.huber_coeff,
                "batch_size" : self.batch_size,
                "lr" : self.lr,
                "self.min_lr" : self.min_lr,
                "average_embeddings" : self.average_embeddings,
                "model_name" : self.model_name,
                "augmentation" : self.augmentation,
                "use_reconstruction" : self.use_reconstruction,
                "re1_coeff" : self.re1_coeff,
                "alg_coeff" : self.alg_coeff,
                "sed_coeff" : self.sed_coeff,
                "plots_path" : self.plots_path,
                "use_conv" : self.use_conv,
                "hidden_size" : self.hidden_size,
                "num_patches" : self.num_patches,
                'epoch' : epoch,
                "frozen_layers" : self.frozen_layers,
                "frozen_high_layers" : self.frozen_high_layers,
                "all_train_loss" : self.all_train_loss, 
                "all_val_loss" : self.all_val_loss, 
                "all_train_mae" : self.all_train_mae, 
                "all_val_mae" : self.all_val_mae, 
                "all_algebraic_pred" : self.all_algebraic_pred, 
                "all_RE1_pred" : self.all_RE1_pred, 
                "all_SED_pred" : self.all_SED_pred, 
                "all_val_algebraic_pred" : self.all_val_algebraic_pred, 
                "all_val_RE1_pred" : self.all_val_RE1_pred, 
                "all_val_SED_pred" : self.all_val_SED_pred
            }, model_path) 

    def load_model(self, parent_model_path=None, continue_training=True):
        model_path = os.path.join(parent_model_path, "model.pth")
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
            except Exception as e:
                backup_path = os.path.join(parent_model_path, "backup_model.pth")
                checkpoint = torch.load(backup_path, map_location='cpu')
                print_and_write(f'\n#########\nusing backup:\n{e}\n', self.plots_path)
                sys.stdout.flush()
        else:
            print_and_write(f"Model {model_path} not found", self.plots_path)
            raise FileNotFoundError

        self.L2_coeff = checkpoint.get("L2_coeff", self.L2_coeff)
        self.huber_coeff = checkpoint.get("huber_coeff", self.huber_coeff)
        self.batch_size = checkpoint.get("batch_size", self.batch_size)
        self.lr = checkpoint.get("lr", self.lr)
        self.min_lr = checkpoint.get("min_lr", self.min_lr)
        self.average_embeddings = checkpoint.get("average_embeddings", self.average_embeddings)
        self.model_name = checkpoint.get("model_name", self.model_name)
        self.augmentation = checkpoint.get("augmentation", self.augmentation)
        self.use_reconstruction = checkpoint.get("use_reconstruction", self.use_reconstruction)
        self.re1_coeff = checkpoint.get("re1_coeff", self.re1_coeff)
        self.alg_coeff = checkpoint.get("alg_coeff", self.alg_coeff)
        self.sed_coeff = checkpoint.get("sed_coeff", self.sed_coeff)
        self.plots_path = checkpoint.get("plots_path", None) if GET_OLD_PATH else self.plots_path 
        self.use_conv = checkpoint.get("use_conv", self.use_conv)
        self.hidden_size = checkpoint.get("hidden_size", 0)
        self.num_patches = checkpoint.get("num_patches", 0)
        self.frozen_layers = checkpoint.get("frozen_layers", self.frozen_layers)
        self.frozen_high_layers = checkpoint.get("frozen_high_layers", 0)
        self.start_epoch = checkpoint.get("epoch", 0) if continue_training else 0
        self.all_train_loss = checkpoint.get("all_train_loss", []) if continue_training else []
        self.all_val_loss = checkpoint.get("all_val_loss", []) if continue_training else []
        self.all_train_mae = checkpoint.get("all_train_mae", []) if continue_training else []
        self.all_val_mae = checkpoint.get("all_val_mae", []) if continue_training else []
        self.all_algebraic_pred = checkpoint.get("all_algebraic_pred", []) if continue_training else []
        self.all_RE1_pred = checkpoint.get("all_RE1_pred", []) if continue_training else []
        self.all_SED_pred = checkpoint.get("all_SED_pred", []) if continue_training else []
        self.all_val_algebraic_pred = checkpoint.get("all_val_algebraic_pred", []) if continue_training else []
        self.all_val_RE1_pred = checkpoint.get("all_val_RE1_pred", []) if continue_training else []
        self.all_val_SED_pred = checkpoint.get("all_val_SED_pred", []) if continue_training else []

        # Initialize loss functions
        self.L2_loss = nn.MSELoss().to(device)
        self.huber_loss = nn.HuberLoss().to(device)

        # Get input dimension for the MLP based on ViT configuration
        self.hidden_size = self.model.config.hidden_sizes[-1] if self.resnet else 1280 if self.model_name==EFFICIENTNET else self.model.config.hidden_size
        self.num_patches = 7 if self.resnet or self.model_name==EFFICIENTNET else self.model.config.image_size // self.model.config.patch_size   
                
        if self.use_conv:
            convnet_input_dim = 1 if self.cc else 2*self.hidden_size 
            self.conv = ConvNet(input_dim=convnet_input_dim, batch_size=self.batch_size).to(device)
            mlp_input_shape = 2 * self.conv.hidden_dims[-1] * MAX_POOL_SIZE**2 
            self.conv.load_state_dict(checkpoint['conv'])
            self.conv.to(device)

        # Load MLP
        self.mlp = MLP(input_dim=mlp_input_shape).to(device)
        self.mlp.load_state_dict(checkpoint['mlp']) 
        self.mlp.to(device)

        # Load model
        self.model.load_state_dict(checkpoint['vit']) 
        self.model.to(device)
        try:
            # Load optimizer and scheduler
            self.optimizer = optim.Adam([
                {'params': self.model.parameters(), 'lr': self.lr},
                {'params': self.mlp.parameters(), 'lr': self.lr},   # Potentially higher learning rate for the MLP
                {'params': self.conv.parameters(), 'lr': self.lr} if self.use_conv else {'params': []}
            ])
            self.scheduler = None
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            self.optimizer = optim.Adam([
                {'params': self.model.parameters(), 'lr': self.lr},
                {'params': []},
                {'params': self.mlp.parameters(), 'lr': self.lr},   # Potentially higher learning rate for the MLP
                {'params': self.conv.parameters(), 'lr': self.lr} if self.use_conv else {'params': []}
            ])
            self.scheduler = None
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
    def append_epoch_stats(self, train_mae, val_mae, epoch_stats):
        self.all_train_mae.append(train_mae)
        self.all_train_loss.append(epoch_stats["loss"])
        self.all_algebraic_pred.append(epoch_stats["algebraic_pred"])  
        self.all_RE1_pred.append(epoch_stats["RE1_pred"])
        self.all_SED_pred.append(epoch_stats["SED_pred"])

        self.all_val_mae.append(val_mae)
        self.all_val_loss.append(epoch_stats["val_loss"])
        self.all_val_algebraic_pred.append(epoch_stats["val_algebraic_pred"])
        self.all_val_RE1_pred.append(epoch_stats["val_RE1_pred"])
        self.all_val_SED_pred.append(epoch_stats["val_SED_pred"])

    def plot_all(self):
        try:
            plot(x=range(1, self.num_epochs + 1), y1=self.all_train_loss, y2=self.all_val_loss, title="Loss", plots_path=self.plots_path)
            plot(x=range(1, self.num_epochs + 1), y1=self.all_train_mae, y2=self.all_val_mae, title="MAE", plots_path=self.plots_path)
            plot(x=range(1, self.num_epochs + 1), y1=self.all_algebraic_pred, y2=self.all_val_algebraic_pred, title="Algebraic_distance", plots_path=self.plots_path)
            plot(x=range(1, self.num_epochs + 1), y1=self.all_RE1_pred, y2=self.all_val_RE1_pred, title="RE1_distance", plots_path=self.plots_path)
            plot(x=range(1, self.num_epochs + 1), y1=self.all_SED_pred, y2=self.all_val_SED_pred, title="SED_distance", plots_path=self.plots_path)
        except Exception as e:
            print_and_write(f"Error plotting: \n{e}", self.plots_path)

    def test(self, test_loader, write=True):
        with torch.no_grad():
            loss, mae, alg, re1, sed, trimmed_algs, trimmed_seds, trimmed_re1s = 0, 0, 0, 0, 0, 0, 0, 0
            for epoch in range(10):
                epoch_stats = {"test_algebraic_pred": torch.tensor(0), "test_algebraic_sqr_pred": torch.tensor(0), "test_RE1_pred": torch.tensor(0), "test_SED_pred": torch.tensor(0), 
                                "test_algebraic_truth": torch.tensor(0), "test_algebraic_sqr_truth": torch.tensor(0), "test_RE1_truth": torch.tensor(0), "test_SED_truth": torch.tensor(0), 
                                "test_loss": torch.tensor(0), "test_labels": torch.tensor([]), "test_outputs": torch.tensor([])}
                send_to_device(epoch_stats)
    
                batch_alg_preds, batch_SED_preds, batch_RE1_preds = self.dataloader_step(test_loader, 0, epoch_stats, data_type="test")

                divide_by_dataloader(epoch_stats, len_test_loader=len(test_loader))

                test_mae = torch.mean(torch.abs(epoch_stats["test_labels"] - epoch_stats["test_outputs"]))

                loss += epoch_stats["test_loss"]
                mae += test_mae.cpu().item()
                alg += epoch_stats["test_algebraic_pred"]
                re1 += epoch_stats["test_RE1_pred"]
                sed += epoch_stats["test_SED_pred"]

#################################################################################################################
                sorted_algs, sorted_seds, sorted_re1s = sorted(batch_alg_preds, key=lambda x: x.cpu().item()), sorted(batch_SED_preds, key=lambda x: x.cpu().item()), sorted(batch_RE1_preds, key=lambda x: x.cpu().item())
                trimmed_alg, trimmed_sed, trimmed_re1 = sorted_algs[:int(len(sorted_algs) * 0.95)], sorted_seds[:int(len(sorted_seds) * 0.95)], sorted_re1s[:int(len(sorted_re1s) * 0.95)]
                trimmed_alg, trimmed_sed, trimmed_re1 = [x.cpu().numpy() for x in trimmed_alg], [x.cpu().numpy() for x in trimmed_sed], [x.cpu().numpy() for x in trimmed_re1]

                trimmed_algs += np.mean(trimmed_alg)
                trimmed_seds += np.mean(trimmed_sed)
                trimmed_re1s += np.mean(trimmed_re1)

                # # Define your bins
                # bins = np.arange(0, 20.4, 0.4).tolist() + [float('inf')]
                # # Plot histogram
                # plt.figure(figsize=(10, 5))
                # plt.hist(batch_SED_preds, bins=bins, edgecolor='black')
                # plt.title('Histogram of batch SED predictions')
                # plt.xlabel('SED Prediction')
                # plt.ylabel('Frequency')
                # file_name = 'kitti_clip_sed_trim_pts' if TRIM_PTS else 'kitti_clip_sed' 
                # plt.savefig(file_name)

                # # Plot histogram
                # plt.figure(figsize=(10, 5))
                # plt.hist(trimmed_seds, bins=bins, edgecolor='black')
                # plt.title('Histogram of batch SED predictions')
                # plt.xlabel('SED Prediction')
                # plt.ylabel('Frequency')
                # file_name = 'kitti_clip_sed_imgs_trimmed_trim_pts' if TRIM_PTS else 'kitti_clip_sed_imgs_trimmed'
                # plt.savefig(file_name)
#################################################################################################################
                 
        output = f"""\n\n## TEST RESULTS: ##
Test Loss: {loss/10}\t\t Test MAE: {mae/10}
Test Algebraic dist: {alg/10}
Test SED dist: {sed/10}
Test RE1 dist: {re1/10}

Trimmed algs: {trimmed_algs/10}
Trimmed seds: {trimmed_seds/10}
Trimmed re1s: {trimmed_re1s/10}

Test Algebraic dist truth: {epoch_stats["test_algebraic_truth"]}
Test SED dist truth: {epoch_stats["test_SED_truth"]}
Test RE1 dist truth: {epoch_stats["test_RE1_truth"]}

{self.plots_path}\n"""
        print_and_write(output, self.plots_path) if write else print(output)
