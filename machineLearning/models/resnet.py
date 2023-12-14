# code for resnet model
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet(nn.Module):
    """
    ResNet is based on ResNet50
    Change num_channels to 3 if using RGB images
    """
    def __init__(self, num_channels):
        self.num_channels = num_channels
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(ResNet, self).__init__()
    
        resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Modify the first convolutional layer to accept one channel instead of three
        resnet_model.conv1 = torch.nn.Conv2d(self.num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet_model.fc = nn.Linear(2048, 1024)

        output_head = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.base_model = resnet_model.to(self.device)
        self.output_head = output_head.to(self.device)
    
    def forward(self, x):
        x = x.to(self.device)
        base_prediction = self.base_model(x)
        output_prediction = self.output_head(base_prediction)
        return output_prediction
    