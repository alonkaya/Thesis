from pykitti.odometry import odometry

from FunMatrix import *
from utils import *
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms import functional as TF

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform, val, seq_name, jump_frames=JUMP_FRAMES):
        self.dataset = dataset
        self.transform = transform
        self.val = val
        self.seq_name = seq_name
        self.jump_frames = jump_frames

    def __len__(self):
        return len(self.dataset.poses)-self.jump_frames if not self.val else min(VAL_LENGTH, len(self.dataset.poses)-self.jump_frames)

    def __getitem__(self, idx):
        first_image = self.dataset.get_cam1(idx) if self.val else self.dataset.get_cam0(idx)
        second_image = self.dataset.get_cam1(idx+self.jump_frames) if self.val else self.dataset.get_cam0(idx+self.jump_frames)
        
        k = torch.tensor(self.dataset.calib.K_cam1, dtype=torch.float32) if self.val else torch.tensor(self.dataset.calib.K_cam0, dtype=torch.float32)

        first_image = self.transform(first_image)
        second_image = self.transform(second_image)
        
        unnormalized_F = get_F(self.dataset.poses, idx, k, k, self.jump_frames)

        # Normalize F-Matrix
        F = norm_layer(unnormalized_F.view(-1, 9)).view(3,3)

        return first_image, second_image, F, self.seq_name
    
transform = v2.Compose([
    # v2.Resize((256, 256), antialias=True),
    # v2.CenterCrop(224),
    # v2.Grayscale(num_output_channels=3),
    v2.ToTensor(),  # Converts to torch.float32 and scales [0,255] -> [0,1]
    # v2.Normalize(mean=norm_mean,  # Normalize each channel
                        # std=norm_std),
])


def get_dataloaders_KITTI2(batch_size=BATCH_SIZE):

    dataset = odometry(base_path='.', sequence='00')

    # Split the dataset based on the calculated samples. Get 00 and 01 as val and the rest as train sets.
    dataset_cam0 = Dataset(dataset, transform, seq_name= f'00', val=False)
    dataset_cam1 = Dataset(dataset, transform, seq_name= f'00', val=True)

    # Create a DataLoader
    train_loader = DataLoader(dataset_cam0, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(dataset_cam1, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader

def please():
    dataset = odometry(base_path='.', sequence='00')
    img_num = 1000
    jump_frames = 2
    for img_num in range(100):
        img1 = dataset.get_cam0(img_num)
        img2 = dataset.get_cam0(img_num  + jump_frames)
        # img2 = dataset.get_cam1(img_num)
        
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)

        k0 = torch.tensor(dataset.calib.K_cam0, dtype=torch.float32)
        k1 = torch.tensor(dataset.calib.K_cam1, dtype=torch.float32)

        unnormalized_F = get_F(dataset.poses, img_num, k0, k0, jump_frames=jump_frames)

        # Normalize F-Matrix
        F = norm_layer(unnormalized_F.view(-1, 9)).view(3,3)

        epipolar_geo = EpipolarGeometry(img1, img2, F)
        sed = epipolar_geo.visualize(idx=img_num, epipolar_lines_path=os.path.join("gt_epipole_lines_KITTI_cam0_take4", '00'))
        print(sed)

if __name__ == "__main__":
    please()