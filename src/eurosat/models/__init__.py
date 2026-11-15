"""One implementation file per backbone; the runner keeps one model resident at a time."""
from importlib import import_module
from ..config import MODEL_IDS

EMBEDDING_DIMENSIONS = {'resnet18': 512, 'convnext_tiny': 768, 'vit_b_16': 768, 'dinov2_vits14': 384}


def load_backbone(name, device, *, pretrained=True):
    if name not in MODEL_IDS:
        raise ValueError('Unknown backbone: ' + name)
    if not pretrained and name != 'dinov2_vits14':
        raise ValueError('Random initialization is supported for DINOv2 only')
    adapter = import_module(f'{__name__}.{name}')
    model = adapter.build(pretrained=pretrained) if name == 'dinov2_vits14' else adapter.build()
    model.requires_grad_(False)
    return model.to(device).eval()
