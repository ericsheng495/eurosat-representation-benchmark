"""Pinned official DINOv2 ViT-S/14 returning the normalized class-token embedding."""
import os
import torch
from ..config import DINO_REVISION


def build():
    os.environ['XFORMERS_DISABLED'] = '1'
    return torch.hub.load('facebookresearch/dinov2:' + DINO_REVISION,
                          'dinov2_vits14', pretrained=True, trust_repo=True, skip_validation=True)
