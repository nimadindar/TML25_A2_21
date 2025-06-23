import torch
import torch.nn as nn
from torchvision import models

class StudentEncoder(nn.Module):
    def __init__(self, model_name, output_dim=1024):
        super().__init__()
        if model_name != "resnet18":
            raise ValueError("Only resnet18 is supported for this implementation.")
        
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        self.feature = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Linear(512, output_dim)

    def forward(self, x):
        feat = self.feature(x).flatten(1)
        return self.head(feat)