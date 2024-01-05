# from FMatrixRegressor import FMatrixRegressor
from params import *
from Dataset import train_loader, val_loader
import torch
from FunMatrix import check_epipolar_constraint
import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# model = FMatrixRegressor(mlp_hidden_sizes, num_output, pretrained_model_name=clip_model_name, lr=learning_rate, device=device, freeze_pretrained_model=False)
# model = model.to(device)

# print(f'learning_rate: {learning_rate}, mlp_hidden_sizes: {mlp_hidden_sizes}, num_of_frames: {num_of_frames}, jump_frames: {jump_frames},  add_penalty_loss: {add_penalty_loss}, enforce_fundamental_constraint: {enforce_fundamental_constraint}')
# model.train_model(train_loader, val_loader, num_epochs=num_epochs)


train_iter = iter(train_loader)
first_image, second_image, label = next(train_iter)

avg_ec_err = 0
for img_1, img_2, F in zip(first_image, second_image, label):
    print(check_epipolar_constraint(img_1, img_2, F))
    
print(avg_ec_err)