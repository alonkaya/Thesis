from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from params import *
import torch.nn as nn
from transformers import CLIPVisionModel
from torchvision import transforms

# Define the transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean, std=norm_std)
])


class ImageFeatureTransformer(nn.Module):
    def __init__(self, patch):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
        self.patch = patch

    def forward(self, x1, x2):
        # Extract image embeddings
        x1_embeddings = self.model(x1).last_hidden_state[:, 1:, :]  # Remove CLS token
        x2_embeddings = self.model(x2).last_hidden_state[:, 1:, :]  # Remove CLS token

        p1 = x1_embeddings[0, self.patch, :].view(1,-1) # shape [1,768]
        attention_map = p1.matmul(x2_embeddings[0].transpose(0,1)) # shape [1, 49]
        return attention_map


    def visualize_attention(self, image1, image2):
        with torch.no_grad():
            attention_map = self.forward(image1, image2).reshape(7,7).numpy()

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.imshow(attention_map, cmap='viridis', aspect='equal')  
        plt.colorbar()
        plt.title('Patch Similarity Map')
        plt.xlabel('Patch Index (Image 2)')
        plt.ylabel('First Patch (Image 1)')

        # Save and show
        plt.savefig(f'attention_map_patch{self.patch}.png')
        # plt.show()


if __name__ == '__main__':
    img1 = Image.open('sequences/00/image_0/000088.png').convert('RGB')
    img2 = Image.open('sequences/00/image_1/000088.png').convert('RGB')
    img1 = transform(img1).unsqueeze(0)
    img2 = transform(img2).unsqueeze(0)

    # img1_np = np.array(img1)
    # img2_np = np.array(img2)
    # fig, axes = plt.subplots(1, 2, figsize=(14, 7))  # Create a figure with 1 row and 2 columns of subplots
    # for ax, img in zip(axes, [img1_np, img2_np]):
    #     ax.imshow(img)
        
    #     # Set grid
    #     ax.set_xticks(np.linspace(0, img.shape[1], 8))  # 16 vertical lines (17 tick positions)
    #     ax.set_yticks(np.linspace(0, img.shape[0], 8))  # 16 horizontal lines (17 tick positions)
    #     ax.grid(color='white', linestyle='--', linewidth=0.7)

    #     # Remove axes labels
    #     ax.set_xticklabels([])
    #     ax.set_yticklabels([])
    # plt.tight_layout()
    # plt.savefig('images.png')


    for i in range(49):
        model = ImageFeatureTransformer(i)
        model.visualize_attention(img1, img2)
