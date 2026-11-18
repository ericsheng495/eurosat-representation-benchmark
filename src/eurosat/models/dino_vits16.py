"""DINO ViT-S/16: a Vision Transformer trained on ImageNet-1K without labels (self-supervised).

Official model from the pinned facebookresearch/dino repository.
Embedding: the final normalized class token (384-d). The official model has no classification head.
"""
import torch

from ..config import DINO_REVISION, MODEL_IDS

CHECKPOINT = MODEL_IDS['dino_vits16']
EMBEDDING_DIM = 384


def backbone(pretrained):
    """The network without a classification head, so it returns embeddings.

    pretrained=False gives random weights (the caller fixes the seed).
    """
    return torch.hub.load('facebookresearch/dino:' + DINO_REVISION, 'dino_vits16', pretrained=pretrained,
                          trust_repo=True, skip_validation=True, verbose=False)
