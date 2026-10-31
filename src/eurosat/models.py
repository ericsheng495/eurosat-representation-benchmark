"""Frozen pretrained CNN and Transformer representation adapters."""
import os
import torch
from torch import nn
from torchvision import models
from .config import DINO_REVISION


def load_backbone(name, device):
    if name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
    elif name == 'convnext_tiny':
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Identity()
    elif name == 'vit_b_16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = nn.Identity()
    elif name == 'dinov2_vits14':
        # Official implementation + weights pinned to a specific source revision.
        os.environ['XFORMERS_DISABLED'] = '1'
        model = torch.hub.load('facebookresearch/dinov2:' + DINO_REVISION,
                               'dinov2_vits14', pretrained=True,
                               trust_repo=True, skip_validation=True)
    else:
        raise ValueError('Unknown backbone: ' + name)
    model.requires_grad_(False)
    return model.to(device).eval()
