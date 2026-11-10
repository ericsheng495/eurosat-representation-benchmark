"""convnext_tiny: pretrained weights and the final embedding extraction point."""
from torch import nn
from torchvision import models


def build():
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier[-1] = nn.Identity()
    return model
