"""ConvNeXt-Tiny: a modern CNN.

Embedding: pooled features after the final LayerNorm and flatten, before the last Linear layer (768-d).
Stochastic depth (torchvision default 0.1) is turned off, so fine-tuning uses no random regularizer,
like the other three models. It only acts in training mode, so embeddings are not affected.
"""
from torch import nn
from torchvision import models

from ..config import MODEL_IDS

CHECKPOINT = MODEL_IDS['convnext_tiny']
EMBEDDING_DIM = 768


def backbone(pretrained):
    """The network without its ImageNet head, so it returns embeddings.

    pretrained=False gives random weights (the caller fixes the seed).
    """
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None,
                                 stochastic_depth_prob=0.0)
    model.classifier[2] = nn.Identity()  # classifier = [LayerNorm2d, Flatten, Linear]; keep the first two.
    return model
