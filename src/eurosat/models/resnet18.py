"""ResNet-18: the standard CNN baseline.

Embedding: global-average-pooled features before the classification head (512-d).
"""
from torch import nn
from torchvision import models

from ..config import MODEL_IDS

CHECKPOINT = MODEL_IDS['resnet18']
EMBEDDING_DIM = 512


def backbone(pretrained):
    """The network without its ImageNet head, so it returns embeddings.

    pretrained=False gives random weights (the caller fixes the seed).
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Identity()
    return model
