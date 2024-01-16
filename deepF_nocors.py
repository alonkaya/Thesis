import torch
import torch.nn as nn
import torch.nn.functional as F

# from UCN.networks import UniversalCorrepondenceNetwork

class HomographyNet(nn.Module):
    def __init__(self, dim=64, ksize=3, use_bn=True, use_dropout=True,
                 out_dim=9, weight_decay=0.01, use_idx=True, use_coor=False,
                 use_reconstruction_module=True):
        super(HomographyNet, self).__init__()
        self.name = "homography_net"
        self.dim = dim
        self.ksize = ksize
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.weight_decay = weight_decay
        self.use_idx = use_idx
        self.use_coor = use_coor
        self.use_reconstruction_module = use_reconstruction_module
        print("HomographNet Use coord:%s" % self.use_coor)
        if self.use_reconstruction_module:
            self.out_dim = 8
        else:
            self.out_dim = out_dim

    def normalize_output(self, x):
        return x / (torch.max(x.view(-1, 9), dim=1) + 1e-8)

    def conv2d(self, x, dim, ksizes, strides, padding):
        conv_sequential = nn.Sequential(
            nn.Conv2d(x.size(1), dim, ksizes, stride=strides, padding=padding),
        )
        return conv_sequential(x)

    def fetch_idx(self, orig_idx, new_idx):
        c = new_idx.size(-1)
        new_idx = float(new_idx / c)
        out = orig_idx.view(-1)[new_idx.view(-1)]
        print("Orig:%s\tNew:%s\tOut:%s" % (orig_idx.shape, new_idx.shape, out.shape))
        return out

    def reconstruction_module(self, x):
        def get_rotation(x):
            R_x = x.clone()
            R_y = x.clone()
            R_z = x.clone()

            R_x[0][0] = 1.0
            R_x[0][1] = 0.0
            R_x[0][2] = 0.0
            R_x[1][0] = 0.0
            R_x[1][1] = torch.cos(x[2])
            R_x[1][2] = -torch.sin(x[2])
            R_x[2][0] = 0.0
            R_x[2][1] = torch.sin(x[2])
            R_x[2][2] = torch.cos(x[2])

            R_y[0][0] = torch.cos(x[3])
            R_y[0][1] = 0.0
            R_y[0][2] = -torch.sin(x[3])
            R_y[1][0] = 0.0
            R_y[1][1] = 1.0
            R_y[1][2] = 0.0
            R_y[2][0] = torch.sin(x[3])
            R_y[2][1] = 0.0
            R_y[2][2] = torch.cos(x[3])

            R_z[0][0] = torch.cos(x[4])
            R_z[0][1] = -torch.sin(x[4])
            R_z[0][2] = 0.0
            R_z[1][0] = torch.sin(x[4])
            R_z[1][1] = torch.cos(x[4])
            R_z[1][2] = 0.0
            R_z[2][0] = 0.0
            R_z[2][1] = 0.0
            R_z[2][2] = 1.0

            R = torch.matmul(R_x, torch.matmul(R_y, R_z))
            return R

        def get_inv_intrinsic(x, i):
            x = x.clone()

            x[0][0] = -1 / (x[i] + 1e-8)
            x[0][1] = 0.0
            x[0][2] = 0.0
            x[1][0] = 0.0
            x[1][1] = -1 / (x[i] + 1e-8)
            x[1][2] = 0.0
            x[2][0] = 0.0
            x[2][1] = 0.0
            x[2][2] = 1.0

            return x
        
        def get_translate(x):
            x = x.clone()

            x[0][0] = 0
            x[0][1] = -x[7]
            x[0][2] = x[6]
            x[1][0] = x[7]
            x[1][1] = 0
            x[1][2] = -x[5]
            x[2][0] = -x[6]
            x[2][1] = x[5]
            x[2][2] = 0

            return x


        def get_fmat(x):
            K1_inv = get_inv_intrinsic(x, 0)
            K2_inv = get_inv_intrinsic(x, 1)
            R = get_rotation(x[2], x[3], x[4])
            T = get_translate(x[5], x[6], x[7])
            F = torch.matmul(K2_inv, torch.matmul(R, torch.matmul(T, K1_inv)))
            flat = F.view(-1)
            return flat

        out = torch.stack([get_fmat(xi) for xi in x])

        return out

    def forward(self, x1, x2, reuse=False):
        if reuse:
            with torch.no_grad():
                x1 = x1.detach()
                x2 = x2.detach()

        print(x1.size())
        print(torch.numel(x1))

        x = torch.cat([x1, x2], dim=3)

        def get_grid(_):
            ret = torch.arange(x.size(1) * x.size(2))
            return ret

        x_idx = torch.stack([get_grid(x) for x in range(x.size(0))])
        print(x_idx.size())

        # Group 1 (128x128)
        conv1_1 = self.conv2d(x, self.dim, self.ksize, 1, padding=torch.ceil((self.ksize - 1) / 2))
        if self.use_bn:
            conv1_1 = nn.BatchNorm2d(self.dim)(conv1_1)
        conv1_1 = F.relu(conv1_1)

        conv1_2 = self.conv2d(conv1_1, self.dim, self.ksize, 1, padding=torch.ceil((self.ksize - 1) / 2))
        if self.use_bn:
            conv1_2 = nn.BatchNorm2d(self.dim)(conv1_2)
        conv1_2 = F.relu(conv1_2)

        conv1, conv1_idx = F.max_pool2d(conv1_2, 4, 4, padding=0, return_indices=True)
        conv1_idx = self.fetch_idx(x_idx, conv1_idx)

        # Group 2 (64x64)
        conv2_1 = self.conv2d(conv1, self.dim, self.ksize, 1, padding=torch.ceil((self.ksize - 1) / 2))
        if self.use_bn:
            conv2_1 = nn.BatchNorm2d(self.dim)(conv2_1)
        conv2_1 = F.relu(conv2_1)
        conv2_2 = self.conv2d(conv2_1, self.dim, self.ksize, 1, padding=torch.ceil((self.ksize - 1) / 2))
        if self.use_bn:
            conv2_2 = nn.BatchNorm2d(self.dim)(conv2_2)
        conv2_2 = F.relu(conv2_2)

        conv2, conv2_idx = F.max_pool2d(conv2_2, 4, 4, padding=0, return_indices=True)
        conv2_idx = self.fetch_idx(conv1_idx, conv2_idx)
        print(conv2_idx.size())
        print(conv2.size())

        # Group 3 (32x32)
        conv3_1 = self.conv2d(conv2, self.dim * 2, self.ksize, 1, padding=torch.ceil((self.ksize - 1) / 2))
        if self.use_bn:
            conv3_1 = nn.BatchNorm2d(self.dim * 2)(conv3_1)
        conv3_1 = F.relu(conv3_1)
        conv3_2 = self.conv2d(conv3_1, self.dim * 2, self.ksize, 1, padding=torch.ceil((self.ksize - 1) / 2))
        if self.use_bn:
            conv3_2 = nn.BatchNorm2d(self.dim * 2)(conv3_2)
        conv3_2 = F.relu(conv3_2)

        # Group 4 (16x16)
        conv4_1 = self.conv2d(conv3_2, self.dim * 2, self.ksize, 1, padding=torch.ceil((self.ksize - 1) / 2))
        if self.use_bn:
            conv4_1 = nn.BatchNorm2d(self.dim * 2)(conv4_1)
        conv4_1 = F.relu(conv4_1)
        conv4_2 = self.conv2d(conv4_1, self.dim * 2, self.ksize, 1, padding=torch.ceil((self.ksize - 1) / 2))
        if self.use_bn:
            conv4_2 = nn.BatchNorm2d(self.dim * 2)(conv4_2)
        conv4_2 = F.relu(conv4_2)

        # Group 5
        conv5_1 = self.conv2d(conv4_2, self.dim * 2, self.ksize, 1, padding=torch.ceil((self.ksize - 1) / 2))
        if self.use_bn:
            conv5_1 = nn.BatchNorm2d(self.dim * 2)(conv5_1)
        conv5_1 = F.relu(conv5_1)
        conv5_2 = self.conv2d(conv5_1, self.dim * 2, self.ksize, 1, padding=torch.ceil((self.ksize - 1) / 2))
        if self.use_bn:
            conv5_2 = nn.BatchNorm2d(self.dim * 2)(conv5_2)
        conv5_2 = F.relu(conv5_2)

        conv5, conv5_idx = F.max_pool2d(conv5_2, 2, 2, padding=0, return_indices=True)
        conv5_idx = self.fetch_idx(conv1_idx, conv5_idx)
        print(conv5_idx.size())
        print(conv5.size())

        if self.use_coor:
            conv5_x = conv5_idx.float() / x.size(1)
            conv5_y = conv5_idx.float() % x.size(1)
            conv5_x = conv5_x / x.size(1)
            conv5_y = conv5_y / x.size(2)
            conv5 = torch.cat([conv5, conv5_x, conv5_y], dim=1)
            print("Use coordinate:(x,y)")
            print(conv5.size())
        elif self.use_idx:
            conv5_idx = conv5_idx.float()
            conv5_idx = conv5_idx / (x.size(1) * x.size(2))
            conv5 = torch.cat([conv5, conv5_idx], dim=1)
            print("Use idx (0,1) normalized.")
            print(conv5.size())

        flat = conv5.view(conv5.size(0), -1)
        print(flat.size())
        dense1 = nn.Linear(flat.size(1), 1024)(flat)
        if self.use_dropout:
            dense1 = nn.Dropout(0.5)(dense1)
        out = nn.Linear(1024, self.out_dim)(dense1)

        if self.use_reconstruction_module:
            out = self.reconstruction_module(out)

        out = self.normalize_output(out)

        return out

    @property
    def vars(self):
        return [param for name, param in self.named_parameters() if self.name in name]
