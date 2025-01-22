from torchvision import models, transforms
from params import *
from Dataset import get_data_loaders, get_dataloader_stereo
from PIL import Image

weights = models.efficientnet.EfficientNet_V2_M_Weights.DEFAULT
model = models.efficientnet_v2_m(weights=weights).to(device)
model.classifier = torch.nn.Identity()
model.avgpool = torch.nn.Identity()
model.eval()
# Define the transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image = Image.open('sequences/00/image_0/000000.png').convert('RGB')
input_tensor = transform(image).unsqueeze(0)  # Create a mini-batch as expected by the model

features1 = model(input_tensor).reshape(1,1280,7,7)  # Get features
features2 = model.features(input_tensor).reshape(1,1280,7,7)  # Get features
# print(model)

print(torch.equal(features1, features2))
