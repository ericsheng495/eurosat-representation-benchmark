"""vit_b_16: pretrained weights and the final embedding extraction point."""
from torch import nn
from torchvision import models


def build():
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads = nn.Identity()
    return model
