from ViTMLPRegressor import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = ViTMLPRegressor(mlp_hidden_sizes, num_output, pretrained_model_name=clip_model_name, lr=learning_rate, device=device, freeze_pretrained_model=False)
model = model.to(device)

model.train_model(train_loader, val_loader, num_epochs=num_epochs)