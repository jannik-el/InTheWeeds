# code for CNN model
import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    CNN model with 5 convolutional layers and 3 fully connected layers
    Change num_channels to 3 if using RGB images
    """
    def __init__(self, num_channels):
        self.num_channels = num_channels
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(CNN, self).__init__()
    
        cnn_model = nn.Sequential(
            nn.Conv2d(self.num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(512*7*7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.cnn_model = cnn_model.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        prediction = self.cnn_model(x)
        return prediction