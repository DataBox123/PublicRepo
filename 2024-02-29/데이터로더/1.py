import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

class StereoImageDataset(Dataset):
    def __init__(self, left_image_paths, right_image_paths, gt_image_paths, transform=None):
        """
        Args:
            left_image_paths (list): List of file paths to the left images.
            right_image_paths (list): List of file paths to the right images.
            gt_image_paths (list): List of file paths to the ground truth images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.left_image_paths = left_image_paths
        self.right_image_paths = right_image_paths
        self.gt_image_paths = gt_image_paths
        self.transform = transform

    def __len__(self):
        return len(self.gt_image_paths)

    def __getitem__(self, idx):
        left_img = read_image(self.left_image_paths[idx])
        right_img = read_image(self.right_image_paths[idx])
        gt_img = read_image(self.gt_image_paths[idx])
        
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
            gt_img = self.transform(gt_img)
        
        return left_img, right_img, gt_img

# Assuming you have lists of file paths for left, right, and GT images
left_image_paths = ['path/to/left1.jpg', 'path/to/left2.jpg', ...]
right_image_paths = ['path/to/right1.jpg', 'path/to/right2.jpg', ...]
gt_image_paths = ['path/to/gt1.jpg', 'path/to/gt2.jpg', ...]

# Define a transform (if needed)
transform = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create the dataset
stereo_dataset = StereoImageDataset(left_image_paths, right_image_paths, gt_image_paths, transform=transform)

# Create the DataLoader
dataloader = DataLoader(stereo_dataset, batch_size=4, shuffle=True)

# Example: Iterate over the DataLoader
for left_imgs, right_imgs, gt_imgs in dataloader:
    # Here you would typically feed the images into your model
    print(left_imgs.shape, right_imgs.shape, gt_imgs.shape)
