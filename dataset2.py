import random
from pykitti.odometry import odometry
from Dataset import transform
from FunMatrix import *
from utils import *
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import v2
from torchvision.transforms import functional as TF



class Dataset_stereo(torch.utils.data.Dataset):
    def __init__(self, dataset, transform, val, seq_name, R, t):
        self.dataset = dataset
        self.transform = transform
        self.k0 = torch.tensor(dataset.calib.K_cam0, dtype=torch.float32)
        self.k1 = torch.tensor(dataset.calib.K_cam1, dtype=torch.float32)
        self.val = val
        self.seq_name = seq_name
        self.R=R
        self.t=t

    def __len__(self):
        return len(self.dataset.poses) - VAL_LENGTH if not self.val else VAL_LENGTH

    def __getitem__(self, idx):
        img1 = self.dataset.get_cam0(idx)
        img2 = self.dataset.get_cam1(idx)

        k0=self.k0.clone()
        k1=self.k1.clone()

        if RANDOM_CROP:
            img1, img2 = TF.resize(img1, (256, 256), antialias=True), TF.resize(img2, (256, 256), antialias=True)
            top_crop, left_crop = random.randint(0, 32), random.randint(0, 32)
            img1, img2 = TF.crop(img1, top_crop, left_crop, 224, 224), TF.crop(img2, top_crop, left_crop, 224, 224)
            k0 = adjust_k_crop(k0, top_crop, left_crop)
            k1 = adjust_k_crop(k1, top_crop, left_crop)

        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        unnormalized_F = get_F(self.dataset.poses, idx, k0, k1, self.R, self.t)
        
        # Normalize F-Matrix
        F = norm_layer(unnormalized_F.view(-1, 9)).view(3,3)

        epi = EpipolarGeometry(img1, img2, F=F)

        return img1, img2, F, epi.pts1, epi.pts2, self.seq_name
    

def dataloader_stereo(batch_size=BATCH_SIZE):

    train_datasets, val_datasets = [], []
    for seq in range(11):
        if seq not in train_seqeunces and seq not in val_sequences: continue

        dataset = odometry(base_path='.', sequence=f'{seq:02}')
        R_relative = torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype=torch.float32)
        t_relative = torch.tensor([0.54, 0, 0], dtype=torch.float32)
        
        dataset_stereo_train = Dataset_stereo(dataset, transform, False, f'{seq:02}', R_relative, t_relative)
        dataset_stereo_val = Dataset_stereo(dataset, transform, True, f'{seq:02}', R_relative, t_relative)

        if seq in train_seqeunces:
            train_datasets.append(dataset_stereo_train)        
        if seq in val_sequences:
            val_datasets.append(dataset_stereo_val)

    # Concatenate datasets
    concat_train_dataset = ConcatDataset(train_datasets)
    concat_val_dataset = ConcatDataset(val_datasets)

    # Create a DataLoader
    train_loader = DataLoader(concat_train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(concat_val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader

if __name__ == "__main__":
    pass