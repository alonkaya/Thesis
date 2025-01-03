from torch.utils.data import DataLoader
from datasets import load_dataset
import torchvision.transforms.functional as F
from params import *
from utils import *
import torchvision.transforms as v2


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, angle_range=ANGLE_RANGE, shift_range=SHIFT_RANGE, plots_path=None):
        self.dataset = dataset
        self.transform = transform
        self.angle_range = angle_range
        self.shift_range = shift_range
        self.plots_path = plots_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        original_image = self.dataset[idx]
        
        # Transform: Resize, center, grayscale
        original_image = self.transform(original_image)

        # Generate random affine params
        angle, shift_x, shift_y = random.uniform(-self.angle_range, self.angle_range), random.uniform(-self.shift_range, self.shift_range), random.uniform(-self.shift_range, self.shift_range)
        translated_image = F.affine(original_image, angle=angle, translate=(shift_x, shift_y), scale=1, shear=0)
        
        # translated_image, original_image = F.to_tensor(translated_image), F.to_tensor(original_image)
        translated_image, original_image = F.normalize(translated_image, norm_mean, norm_std), F.normalize(original_image, norm_mean, norm_std)

        # Rescale params -> [-1,1]
        angle = 0 if self.angle_range==0 else torch.tensor(angle / self.angle_range, dtype=torch.float32).to(device)
        shift_x = 0 if self.shift_range==0 else torch.tensor(shift_x / self.shift_range, dtype=torch.float32).to(device)
        shift_y = 0 if self.shift_range==0 else torch.tensor(shift_y / self.shift_range, dtype=torch.float32).to(device)

        if SHIFT_RANGE == 0:
            return original_image, translated_image, angle
        elif ANGLE_RANGE == 0:
            return original_image, translated_image, shift_x, shift_y
        else:
            return original_image, translated_image, angle, shift_x, shift_y


def get_dataloaders(batch_size=BATCH_SIZE, train_length=train_length, val_length=val_length, test_length=test_length, plots_path=None):
    transform = v2.Compose([
        v2.Resize(256),
        v2.RandomCrop(224),
        v2.Grayscale(num_output_channels=3),
        v2.ColorJitter(brightness=0.3, contrast=0.3),
        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.35)),
        # v2.ToTensor(),
    ])

    # Load and display the image
    dataset = load_dataset("frgfm/imagenette", "320px")

    train_data = dataset["train"].select(range(train_length))
    val_data = dataset['validation'].select(range(val_length))
    test_data = dataset['validation'].select(range(len(dataset['validation']) - test_length, len(dataset['validation'])))

    train_images, val_images, test_images = [], [], []
    for img in train_data:
        train_images.append(F.to_tensor(img['image'].convert('RGB')).to(device))
    for img in val_data:
        val_images.append(F.to_tensor(img['image'].convert('RGB')).to(device))
    for img in test_data:
        test_images.append(F.to_tensor(img['image'].convert('RGB')).to(device))

    # Create an instance dataset
    custom_train_dataset = CustomDataset(train_images, transform=transform, angle_range=ANGLE_RANGE, shift_range=SHIFT_RANGE, plots_path=plots_path)
    custom_val_dataset = CustomDataset(val_images, transform=transform,angle_range=ANGLE_RANGE, shift_range=SHIFT_RANGE, plots_path=plots_path)
    custom_test_dataset = CustomDataset(test_images, transform=transform,angle_range=ANGLE_RANGE, shift_range=SHIFT_RANGE, plots_path=plots_path)

    # # Create a DataLoader
    train_loader = DataLoader(custom_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=NUM_WORKERS)
    val_loader = DataLoader(custom_val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(custom_test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader



# Function to display original and rotated images using PIL
def show_images(original_image, rotated_image, mean=norm_mean, std=norm_std):
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    
    original_pil_image = F.to_pil_image(original_image * std + mean)
    rotated_pil_image = F.to_pil_image(rotated_image * std + mean)

    # Display the original and rotated images using PIL
    original_pil_image.show(title="Original Image")
    rotated_pil_image.show(title="Rotated Image")

    # Optionally, save the images if needed
    original_pil_image.save("original_image2.png")
    rotated_pil_image.save("rotated_image2.png")

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32, train_length=train_length[0], val_length=val_length, test_length=test_length)
    
    # Get a batch of images
    it = iter(test_loader)
    original_image, rotated_image, _, _, _ = next(it)

    # Visualize the first image in the batch
    show_images(original_image[0], rotated_image[0])  # Display the first image from the batch
