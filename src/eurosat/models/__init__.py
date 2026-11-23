"""Backbones for the three baselines. One file per model; this module builds every baseline the same way.

  untrained   random weights (fixed seed), no training
  pretrained  the official ImageNet-1K weights
  finetuned   ImageNet-1K weights, then trained on the six base classes (weights file from fine-tuning)

Every call builds a new model instance, so no model is ever shared between baselines.
Each backbone returns embeddings: its classification head is removed or was never there.
"""
import hashlib
import json
from importlib import import_module
from pathlib import Path

import torch
from torch import nn

from ..config import BASE_CLASSES, BASELINES, MODEL_IDS, UNTRAINED_SEED

MODELS = list(MODEL_IDS)


def _module(name):
    if name not in MODEL_IDS:
        raise ValueError(f'Unknown model {name!r}; choose from {", ".join(MODELS)}')
    return import_module(f'{__name__}.{name}')


EMBEDDING_DIMENSIONS = {name: _module(name).EMBEDDING_DIM for name in MODELS}


def fine_tuning_network(name):
    """The start of baseline 3: a fresh pretrained backbone + a new head for the six base classes.

    The head starts at zero, so every model starts at the same loss (ln 6) whatever its embedding size.
    A random head would give DINO, whose embeddings are about 5x longer, large first gradients.
    """
    head = nn.Linear(EMBEDDING_DIMENSIONS[name], len(BASE_CLASSES))
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return nn.Sequential(_module(name).backbone(pretrained=True), head)


def finetuned_files(name, weights_dir):
    """Where fine-tuning saves a model: the weights and a JSON record (recipe, history, SHA256)."""
    stem = Path(weights_dir) / f'{name}_finetuned'
    return stem.with_suffix('.pt'), stem.with_suffix('.json')


def _finetuned_record(name, weights_dir):
    if weights_dir is None:
        raise ValueError('The finetuned baseline needs weights_dir (where fine-tuning saved its weights)')
    weights, record = finetuned_files(name, weights_dir)
    if not weights.is_file() or not record.is_file():
        raise FileNotFoundError(f'No fine-tuned {name} in {Path(weights_dir)}: run fine-tuning first '
                                f'(expected {weights.name} and {record.name})')
    return weights, json.loads(record.read_text())


def build_backbone(name, baseline, weights_dir=None):
    """A new, unfrozen backbone instance for one baseline (on the CPU)."""
    if baseline not in BASELINES:
        raise ValueError(f'Unknown baseline {baseline!r}; choose from {", ".join(BASELINES)}')
    module = _module(name)
    if baseline == 'pretrained':
        return module.backbone(pretrained=True)
    if baseline == 'untrained':
        # A fixed seed gives the same random weights every time, without touching the global random state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(UNTRAINED_SEED)
            return module.backbone(pretrained=False)

    weights, record = _finetuned_record(name, weights_dir)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != record['weights_sha256']:
        raise ValueError(f'{weights} does not match the SHA256 in its record: wrong or partly copied file')
    network = nn.Sequential(module.backbone(pretrained=False), nn.Linear(module.EMBEDDING_DIM, len(BASE_CLASSES)))
    network.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    return network[0]  # Drop the six-class head; the backbone returns embeddings.


def load_backbone(name, device, baseline='pretrained', weights_dir=None):
    """A new frozen backbone in evaluation mode, ready to extract embeddings."""
    model = build_backbone(name, baseline, weights_dir)
    model.requires_grad_(False)
    return model.to(device).eval()


def backbone_identity(name, baseline, weights_dir=None):
    """Everything that decides a backbone's embeddings. Used to name caches and to label results."""
    identity = {'model': name, 'baseline': baseline, 'architecture': MODEL_IDS[name]}
    if baseline == 'untrained':
        identity['seed'] = UNTRAINED_SEED
    elif baseline == 'pretrained':
        identity['checkpoint'] = MODEL_IDS[name]
    else:
        identity['weights_sha256'] = _finetuned_record(name, weights_dir)[1]['weights_sha256']
    return identity
