import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomHorizontalFlip

class StereoImageDataset(Dataset):
    def __init__(self, left_image_paths, gt_image_paths, transform=None, gt_transform=None):
        """
        Args:
            left_image_paths (list): List of file paths to the left images.
            gt_image_paths (list): List of file paths to the ground truth images.
            transform (callable, optional): Optional transform to be applied on a sample.
            gt_transform (callable, optional): Optional transform to be applied on the ground truth to create the right image.
        """
        self.left_image_paths = left_image_paths
        self.gt_image_paths = gt_image_paths
        self.transform = transform
        self.gt_transform = gt_transform  # Transformation for gt_img to produce right_img

    def __len__(self):
        return len(self.gt_image_paths)

    def __getitem__(self, idx):
        left_img = read_image(self.left_image_paths[idx])
        gt_img = read_image(self.gt_image_paths[idx])
        
        # Apply the same transformation to left_img and gt_img if specified
        if self.transform:
            left_img = self.transform(left_img)
            gt_img = self.transform(gt_img)
        
        # Transform gt_img to create right_img
        if self.gt_transform:
            right_img = self.gt_transform(gt_img)
        else:
            right_img = gt_img  # No transformation implies right_img is just the gt_img

        return left_img, right_img, gt_img

# Example transformations
transform = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Transformation for gt_img to produce right_img, for example, a random horizontal flip
gt_transform = Compose([
    RandomHorizontalFlip(p=1.0),  # Always apply horizontal flip for demonstration
])

# Assuming you have lists of file paths for left and GT images
left_image_paths = ['path/to/left1.jpg', 'path/to/left2.jpg', ...]
gt_image_paths = ['path/to/gt1.jpg', 'path/to/gt2.jpg', ...]

# Create the dataset with the new gt_transform parameter
stereo_dataset = StereoImageDataset(left_image_paths, gt_image_paths, transform=transform, gt_transform=gt_transform)

# Create the DataLoader
dataloader = DataLoader(stereo_dataset, batch_size=4, shuffle=True)

# Example: Iterate over the DataLoader
for left_imgs, right_imgs, gt_imgs in dataloader:
    # Here you would typically feed the images into your model
    print(left_imgs.shape, right_imgs.shape, gt_imgs.shape)
