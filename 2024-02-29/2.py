import torch
import torch.nn as nn
import torch.nn.functional as F

class StereoImageToSingleImageNet(nn.Module):
    def __init__(self):
        super(StereoImageToSingleImageNet, self).__init__()
        
        # Assuming input stereo images are concatenated along the channel dimension
        # Input shape: [batch_size, 6, 256, 256] for RGB images (3 channels per image)
        
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # Output 3 channels for the RGB image

    def forward(self, x):
        # Apply a series of convolutions and non-linearities to process the stereo images
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)  # Output is a single image
        return x

# Example usage:
# Initialize the model
model = StereoImageToSingleImageNet()

# Example input tensor (batch size, channels, height, width)
# Here, we assume 2 RGB images concatenated along the channel dimension, so 6 channels in total
# For an input image size of 256x256, the input dimension would be [batch_size, 6, 256, 256]
input_tensor = torch.randn(1, 6, 256, 256)

# Forward pass through the model
output_image = model(input_tensor)
print(output_image.shape)  # Expected output shape: [1, 3, 256, 256]
