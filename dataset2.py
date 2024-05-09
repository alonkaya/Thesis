from pykitti.odometry import odometry

from FunMatrix import *
from utils import *
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms import functional as TF

    
# transform = v2.Compose([
#     v2.Resize((256, 256), antialias=True),
#     v2.CenterCrop(224),
#     v2.Grayscale(num_output_channels=3),
#     v2.ToTensor(),  # Converts to torch.float32 and scales [0,255] -> [0,1]
#     v2.Normalize(mean=norm_mean,  # Normalize each channel
#                         std=norm_std),
# ])



def please():
    dataset = odometry(base_path='.', sequence='00')
    img_num = 1000
    jump_frames = 2
    for img_num in range(100):
        img1 = dataset.get_cam0(img_num)
        img2 = dataset.get_cam1(img_num)
        
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)

        k0 = torch.tensor(dataset.calib.K_cam0, dtype=torch.float32)
        k1 = torch.tensor(dataset.calib.K_cam1, dtype=torch.float32)

        unnormalized_F = get_F(dataset.poses, img_num, k0, k1, jump_frames=jump_frames)

        # Normalize F-Matrix
        F = norm_layer(unnormalized_F.view(-1, 9)).view(3,3)

        epipolar_geo = EpipolarGeometry(img1, img2, F)
        sed = epipolar_geo.visualize(idx=img_num, epipolar_lines_path=os.path.join("gt_epipole_lines_KITTI_cam0_take4", '00'))
        print(sed)

if __name__ == "__main__":
    please()