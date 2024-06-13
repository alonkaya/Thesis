import torch
import torch.nn as nn
import torch.nn.functional as F
from params import BATCH_SIZE

class Inception(nn.Module):
    def __init__(self, in_channels, conv11_size, conv33_reduce_size, conv33_size,
                 conv55_reduce_size, conv55_size, pool11_size):
        super(Inception, self).__init__()
        self.conv11 = nn.Conv2d(in_channels, conv11_size, kernel_size=1)
        
        self.conv33_reduce = nn.Conv2d(in_channels, conv33_reduce_size, kernel_size=1)
        self.conv33 = nn.Conv2d(conv33_reduce_size, conv33_size, kernel_size=3, padding=1)
        
        self.conv55_reduce = nn.Conv2d(in_channels, conv55_reduce_size, kernel_size=1)
        self.conv55 = nn.Conv2d(conv55_reduce_size, conv55_size, kernel_size=5, padding=2)
        
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv2d(in_channels, pool11_size, kernel_size=1)

    def forward(self, x):
        conv11 = F.relu(self.conv11(x))
        
        conv33_reduce = F.relu(self.conv33_reduce(x))
        conv33 = F.relu(self.conv33(conv33_reduce))
        
        conv55_reduce = F.relu(self.conv55_reduce(x))
        conv55 = F.relu(self.conv55(conv55_reduce))
        
        max_pool = self.max_pool(x)
        conv_pool = F.relu(self.conv_pool(max_pool))
        
        return torch.cat([conv11, conv33, conv55, conv_pool], 1)

class UniversalCorrespondenceNetwork(nn.Module):
    def __init__(self, in_channels=3):
        super(UniversalCorrespondenceNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_reduce = nn.Conv2d(64, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)

        self.inception_3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv11 = nn.Conv2d(480, 192, kernel_size=1)
        self.conv33 = nn.Conv2d(480, 208, kernel_size=3, padding=1)
        self.conv55 = nn.Conv2d(480, 48, kernel_size=5, padding=2)

        self.pool_proj = nn.Conv2d(480, 64, kernel_size=1)
        self.feature_unnorm = nn.Conv2d(512, 128, kernel_size=1)

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        pool1 = self.pool1(conv1)
        pool1_lrn = F.local_response_norm(pool1, size=5, alpha=0.0001, beta=0.75)

        conv2_reduce = F.relu(self.conv2_reduce(pool1_lrn))
        conv2 = F.relu(self.conv2(conv2_reduce))
        conv2_lrn = F.local_response_norm(conv2, size=5, alpha=0.0001, beta=0.75)
        pool2 = self.pool1(conv2_lrn)

        inception_3a = self.inception_3a(pool2)
        inception_3b = self.inception_3b(inception_3a)
        pool3 = self.pool3(inception_3b)

        conv11 = F.relu(self.conv11(pool3))
        conv33 = F.relu(self.conv33(pool3))
        conv55 = F.relu(self.conv55(pool3))

        pool_proj = F.max_pool2d(pool3, kernel_size=3, stride=1, padding=1)
        pool11 = F.relu(self.pool_proj(pool_proj))

        inception4a = torch.cat([conv11, conv33, conv55, pool11], 1)

        feature_unnorm = self.feature_unnorm(inception4a)
        feature = F.normalize(feature_unnorm, p=2, dim=1)
        return feature
    
class FeatureExtractorDeepF(nn.Module):
    def __init__(self):
        super(FeatureExtractorDeepF, self).__init__()
        self.ucn = UniversalCorrespondenceNetwork() # img shape is (batch_size, 3, 224, 224)

        self.conv_3x3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn_3x3 = nn.BatchNorm2d(128) 
        self.conv_1x1 = nn.Conv2d(128, 128, kernel_size=1)
        self.bn_1x1 = nn.BatchNorm2d(128) 

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.flatten = nn.Flatten()

    def forward(self, x1, x2):
        # UCN
        feature1 = self.ucn(x1) # Output shape is (batch_size, 128, 14, 14)
        feature2 = self.ucn(x2) # Output shape is (batch_size, 128, 14, 14)

        # Concat
        concatenated_features = torch.cat([feature1, feature2], dim=1) # Shape is (batch_size, 256, 14, 14)

        # Convolutional layers after concatenation
        conv3x3 = F.relu(self.bn_3x3(self.conv_3x3(concatenated_features))) # input shape: (batch_size, 256, 14, 14), output shape: (batch_size, 128, 14, 14)
        conv1x1 = F.relu(self.bn_1x1(self.conv_1x1(conv3x3)))               # input shape: (batch_size, 128, 14, 14), output shape: (batch_size, 128, 14, 14)

        # Pooling
        pooled_features, indices = self.pool(conv1x1) # Output shape is (batch_size, 128, 7, 7)
        
        # normalize the indices by dividing each index by the total number of elements in the pooled feature map (i.e. 7 * 7 = 49).
        indices = indices.float() / (pooled_features.shape[2] * pooled_features.shape[3])
        indices = indices.expand_as(pooled_features)
        
        pooled_features_with_position = torch.cat((pooled_features, indices), dim=1) # Output shape: (batch_size, 256, 7, 7)
        pooled_features_with_position = self.flatten(pooled_features_with_position)

        return pooled_features_with_position
        