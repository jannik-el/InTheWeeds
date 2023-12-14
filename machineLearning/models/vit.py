# code for visual transformer model
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTConfig, ViTModel

class ViT(nn.Module):
    """
    ViT based on Vision Transformer architecture with a head for predicting output values
    change num_channels to 3 if using RGB images
    """
    def __init__(self, num_channels):
        self.num_channels = num_channels
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(ViT, self).__init__()
    
        config = ViTConfig(num_channels=self.num_channels, hidden_size=96)
        vit_model = ViTModel(config)

        output_head = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.base_model = vit_model.to(self.device)
        self.output_head = output_head.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        base_prediction = self.base_model(x)
        base_prediction = torch.tensor(base_prediction[1], dtype=torch.float32).to(self.device)
        output_prediction = self.output_head(base_prediction)
        return output_prediction