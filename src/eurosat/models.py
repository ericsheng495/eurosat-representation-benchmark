"""Frozen pretrained CNN and Transformer representation adapters."""
import os
import torch
from torch import nn
from torchvision import models
from .config import DINO_REVISION

FEATURE_DIMENSIONS = {
    'resnet18': 512,
    'convnext_tiny': 768,
    'vit_b_16': 768,
    'dinov2_vits14': 384,
}


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


def extract_features(model, images, name):
    """Return a checked [batch, feature] tensor from a frozen backbone."""
    if name not in FEATURE_DIMENSIONS:
        raise ValueError('Unknown backbone: ' + str(name))
    features = model(images)
    if not isinstance(features, torch.Tensor):
        raise TypeError('Backbone output must be a torch.Tensor')
    features = features.flatten(1)
    expected = FEATURE_DIMENSIONS[name]
    if features.ndim != 2 or features.shape[1] != expected:
        raise ValueError(f'{name} must return {expected} features per image; '
                         f'got shape {tuple(features.shape)}')
    if features.shape[0] != images.shape[0]:
        raise ValueError('Backbone output batch size must match its input')
    return features
