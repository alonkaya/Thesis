from PIL import Image
from Dataset import get_data_loaders
from params import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import  CLIPVisionModel
from torchvision import transforms

# Define the transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class RoMaNet(nn.Module):
    def __init__(self, dino_model, device='cuda'):
        super(RoMaNet, self).__init__()
        self.model = CLIPVisionModel.from_pretrained(dino_model)

        # Assuming an output size similar to the number of patches
        self.num_patches = (224 // 16) * (224 // 16)  # as image size is 224 and patch size is 16
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=768, nhead=12), 
            num_layers=6
        )
        self.anchor_points = nn.Parameter(torch.rand(self.num_patches, 2))  # Randomly initialize anchor points
        self.classifier = nn.Linear(768, self.num_patches)  # Output layer for anchor probabilities

    def forward(self, x1, x2):
        # Extract features
        x1_embeddings = self.model(x1).last_hidden_state[:, 1:, :]  # Eliminate the CLS token
        x2_embeddings = self.model(x2).last_hidden_state[:, 1:, :]  # Eliminate the CLS token

        # Feature matching using transformer decoder
        # For simplicity, assume x1_embeddings as queries and x2_embeddings as keys and values
        query = x1_embeddings.permute(1, 0, 2)  # Shape: [seq_len, batch, features]
        key_value = x2_embeddings.permute(1, 0, 2)  # Shape: [seq_len, batch, features]
        decoded_features = self.transformer_decoder(query, key_value)
        print(decoded_features.shape)
        # Classify probabilities for each anchor point
        anchor_probs = F.softmax(self.classifier(decoded_features), dim=-1)

        # Compute softargmax to get predicted coordinates
        predicted_coordinates = torch.matmul(anchor_probs, self.anchor_points)

        return predicted_coordinates

if __name__ == '__main__':
    img1 = Image.open('sequences/00/image_0/000000.png').convert('RGB')
    img2 = Image.open('sequences/00/image_1/000000.png').convert('RGB')
    img1 = transform(img1).unsqueeze(0)  
    img2 = transform(img2).unsqueeze(0)  

    dino = "facebook/dino-vitb16"
    clip = "openai/clip-vit-base-patch16"

    roma = RoMaNet(clip)
    predictions = roma(img1, img2)
    print(predictions.shape) 































import torch
import torch.nn as nn
import torch.nn.functional as F

class ManualTransformerDecoder(nn.Module):
    def __init__(self, hidden_size, num_heads, num_patches, dropout=0.1):
        super(ManualTransformerDecoder, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.num_patches = num_patches
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Anchor points and classifier remain unchanged
        self.anchor_points = nn.Parameter(torch.rand(self.num_patches, 2))  # Randomly initialize anchor points
        self.classifier = nn.Linear(hidden_size, self.num_patches)  # Output layer for anchor probabilities

    def forward(self, x1_embeddings, x2_embeddings):
        batch_size = x1_embeddings.size(0)
        
        # Project inputs to queries, keys, values
        queries = self.query_proj(x1_embeddings).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key_proj(x2_embeddings).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_proj(x2_embeddings).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Weighted sum of values
        context = torch.matmul(attention_probs, values)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        
        # Final projection
        output = self.output_proj(context) # shape (batch_size, num_patches, hidden_size)
        
        # Classify probabilities for each anchor point and compute softargmax
        anchor_probs = F.softmax(self.classifier(output), dim=-1) # shape (batch_size, num_patches)
        predicted_coordinates = torch.matmul(anchor_probs, self.anchor_points)
        
        return predicted_coordinates

# Example use of the manual Transformer decoder
hidden_size = 768  # typical size for DINO-ViT models
num_heads = 12
num_patches = (224 // 16) * (224 // 16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model and move it to the appropriate device
model = ManualTransformerDecoder(hidden_size, num_heads, num_patches).to(device)











import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        
        Q = self.query(query).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            scores += attn_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V).transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        return self.out(attn_output)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.cross_attn = MultiHeadAttention(d_model, nhead)
        self.ffn = FeedForwardNetwork(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention
        tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return self.norm(tgt)

# Example Usage
d_model = 768
nhead = 12
num_layers = 6

decoder = TransformerDecoder(d_model=d_model, nhead=nhead, num_layers=num_layers)
tgt = torch.randn(10, 20, d_model)  # (batch_size, tgt_seq_len, d_model)
memory = torch.randn(10, 15, d_model)  # (batch_size, mem_seq
