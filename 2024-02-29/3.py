import torch
import torch.nn as nn
import torch.nn.functional as F

class StereoImageProcessor(nn.Module):
    def __init__(self):
        super(StereoImageProcessor, self).__init__()
        
        # Define the shared layers for processing each stereo image
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Layers after merging the features from both images
        self.merge_conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)  # 128 * 2 = 256
        self.merge_conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, left_image, right_image):
        # Process the first image
        x1 = F.relu(self.conv1(left_image))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        
        # Process the second image
        x2 = F.relu(self.conv1(right_image))  # Reuse the same layers for feature extraction
        x2 = F.relu(self.conv2(x2))
        x2 = F.relu(self.conv3(x2))
        
        # Merge the features from both images
        merged = torch.cat((x1, x2), dim=1)  # Concatenate along the channel dimension
        
        # Further processing after merging
        x = F.relu(self.merge_conv1(merged))
        x = F.relu(self.merge_conv2(x))
        x = self.final_conv(x)
        
        return x

# Example usage:
# Initialize the model
model = StereoImageProcessor()

# Example input tensors for left and right images
# Input dimensions: [batch_size, 3, 256, 256] for each RGB image
left_image = torch.randn(1, 3, 256, 256)
right_image = torch.randn(1, 3, 256, 256)

# Forward pass through the model
output_image = model(left_image, right_image)
print(output_image.shape)  # Expected output shape: [1, 3, 256, 256]
