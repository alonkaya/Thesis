from ViTMLPRegressor import *
from params import *
from CustomDataset import *

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders()

    model = ViTMLPRegressor(mlp_hidden_sizes, num_output, pretrained_model_name=clip_model_name, lr=learning_rate, device=device, regress = True, freeze_pretrained_model=False)
    model = model.to(device)

    model.train_model(train_loader, val_loader, num_epochs=num_epochs)