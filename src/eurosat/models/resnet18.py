"""resnet18: pretrained weights and the final embedding extraction point."""
from torch import nn
from torchvision import models


def build():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    return model
