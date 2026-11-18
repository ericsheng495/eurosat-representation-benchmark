"""ViT-B/16: a Vision Transformer trained with ImageNet labels (supervised).

Embedding: the final normalized class token (768-d).
"""
from torch import nn
from torchvision import models

from ..config import MODEL_IDS

CHECKPOINT = MODEL_IDS['vit_b_16']
EMBEDDING_DIM = 768


def backbone(pretrained):
    """The network without its ImageNet head, so it returns embeddings.

    pretrained=False gives random weights (the caller fixes the seed).
    """
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
    model.heads = nn.Identity()
    return model
