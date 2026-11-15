"""DINOv2 architecture trained from random initialization on manifest base classes."""
from dataclasses import asdict, dataclass
import gc
import json
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader

from .cache import atomic, digest_file, key, write_json
from .config import BASE_CLASSES, MODEL_IDS
from .models import EMBEDDING_DIMENSIONS, load_backbone
from .preprocessing.data import load_source
from .preprocessing.images import Images
from .preprocessing.manifest import validate_manifest
from .runtime import environment


@dataclass
class ScratchSettings:
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = .05
    per_class_limit: int | None = None

    @classmethod
    def from_config(cls, config):
        defaults = {'epochs': 1, 'per_class_limit': 2} if config.mode == 'smoke' else {}
        try:
            settings = cls(**(defaults | config.scratch_training))
        except TypeError as error:
            raise ValueError('Unknown scratch-training setting or invalid settings dictionary') from error
        if any(type(v) is not int or v < 1 for v in [settings.epochs, settings.batch_size]):
            raise ValueError('Scratch epochs and batch_size must be positive integers')
        if settings.batch_size < 2:
            raise ValueError('Use batch_size >= 2 for CNN batch normalization')
        if settings.per_class_limit is not None and (type(settings.per_class_limit) is not int or settings.per_class_limit < 2):
            raise ValueError('per_class_limit must be None or an integer >= 2')
        if (not math.isfinite(settings.learning_rate) or settings.learning_rate <= 0 or
                not math.isfinite(settings.weight_decay) or settings.weight_decay < 0):
            raise ValueError('Scratch learning_rate must be positive; weight_decay must be nonnegative')
        return settings


class BaseImages(Images):
    """Reuse the benchmark input pipeline, with reproducible training-only horizontal flips."""
    def __init__(self, root, records, seed, epoch=0, training=False):
        super().__init__(root, records, [(i, 0) for i in range(len(records))], 'none', seed)
        self.epoch, self.training = epoch, training

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        seed = int(key([self.seed, self.records[index]['path'], self.epoch])[:15], 16)
        if self.training and random.Random(seed).random() < .5:
            image = image.flip(-1)
        return image, label


def base_records(source, validated, settings, seed):
    """Read only base train/val pools. Base test and every novel image stay untouched."""
    inventory = {r['path']: r for r in source.records}
    groups = {}
    for phase in ['train', 'val']:
        rows = []
        for label, name in enumerate(BASE_CLASSES):
            filenames = sorted(validated.payload['base'][phase][name])
            if settings.per_class_limit is not None:
                rng = random.Random(int(key([seed, phase, name])[:15], 16))
                filenames = sorted(rng.sample(filenames, min(settings.per_class_limit, len(filenames))))
            rows.extend({**inventory[f'{name}/{f}'], 'label': label} for f in filenames)
        groups[phase] = rows
    if not groups['train'] or not groups['val']:
        raise ValueError('Scratch training requires nonempty base train and validation pools')
    if {r['pixels_sha256'] for r in groups['train']} & {r['pixels_sha256'] for r in groups['val']}:
        raise ValueError('Base train/validation pixels overlap')
    return groups


def _seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _epoch(network, loader, device, optimizer=None):
    network.train(optimizer is not None)
    total_loss = correct = count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits = network(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite scratch training loss')
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        count += len(labels)
    return {'loss': total_loss / count, 'accuracy': correct / count, 'images': count}


def _train(name, source, groups, config, settings, directory, plan_id, device):
    """Checkpoint optimizer/current/best state together at every completed epoch."""
    _seed(config.seed)
    backbone = load_backbone(name, device, pretrained=False).requires_grad_(True)
    network = nn.Sequential(backbone, nn.Linear(EMBEDDING_DIMENSIONS[name], len(BASE_CLASSES)).to(device))
    optimizer = torch.optim.AdamW(network.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    last = directory / 'last.pt'
    start, history, best_state = 0, [], None
    best_accuracy, best_loss, best_epoch = -1., float('inf'), 0
    try:
        if last.exists():
            saved = torch.load(last, map_location='cpu', weights_only=True)
            if saved['plan_id'] != plan_id:
                raise ValueError('Scratch resume checkpoint belongs to another training plan')
            network.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            start, history, best_state = saved['epoch'], saved['history'], saved['best_state']
            best_accuracy, best_loss, best_epoch = saved['best_accuracy'], saved['best_loss'], saved['best_epoch']
            del saved
            print(f'CACHE HIT scratch resume {name}: {start}/{settings.epochs} epochs', flush=True)

        for epoch in range(start, settings.epochs):
            # Epoch-local seeds make an interrupted epoch repeatable on the same stack/device.
            _seed(config.seed + epoch)
            train = BaseImages(source.root, groups['train'], config.seed, epoch, training=True)
            valid = BaseImages(source.root, groups['val'], config.seed)
            # Avoid a final one-image training batch with BatchNorm; keep all images.
            batch_size = min(settings.batch_size, len(train))
            while batch_size > 2 and len(train) % batch_size == 1:
                batch_size -= 1
            if len(train) % batch_size == 1:
                batch_size = len(train)
            loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(config.seed + epoch))
            train_metrics = _epoch(network, loader, device, optimizer)
            val_metrics = _epoch(network, DataLoader(valid, batch_size=settings.batch_size, num_workers=0), device)
            history.append({'epoch': epoch + 1, 'train': train_metrics, 'val': val_metrics})
            if (val_metrics['accuracy'], -val_metrics['loss']) > (best_accuracy, -best_loss):
                best_accuracy, best_loss, best_epoch = val_metrics['accuracy'], val_metrics['loss'], epoch + 1
                best_state = _cpu_state(backbone)
            state = {'plan_id': plan_id, 'epoch': epoch + 1, 'model': _cpu_state(network),
                     'optimizer': optimizer.state_dict(), 'history': history, 'best_state': best_state,
                     'best_accuracy': best_accuracy, 'best_loss': best_loss, 'best_epoch': best_epoch}
            atomic(last, lambda p: torch.save(state, p))
            del state
            print(f'SCRATCH {name} epoch {epoch + 1}/{settings.epochs}: '
                  f'train={train_metrics["accuracy"]:.3f}, base val={val_metrics["accuracy"]:.3f}', flush=True)

        atomic(directory / 'best.pt', lambda p: torch.save(best_state, p))
        return {'best_epoch': best_epoch, 'base_val_accuracy': best_accuracy, 'history': history}
    finally:
        del network, backbone, optimizer
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()


def prepare_scratch_checkpoints(config, storage, scratch, manifest_path, device):
    """Train DINOv2 only; publish an identity from its actual selected weights."""
    config.validate()
    settings = ScratchSettings.from_config(config)
    source = load_source(storage, scratch, config.seed)
    validated = validate_manifest(manifest_path, source)
    groups = base_records(source, validated, settings, config.seed)
    artifacts = {}
    for name in ['dinov2_vits14']:
        plan = {'version': 'supervised-base-scratch-v1', 'model': name, 'architecture_source': MODEL_IDS[name],
                'initialization': 'random', 'archive_sha256': source.archive_sha256,
                'manifest_sha256': validated.sha256, 'seed': config.seed, 'settings': asdict(settings),
                'mode': config.mode, 'versions': environment(), 'training_device': str(device),
                'preprocessing': 'rgb-bilinear224-imagenet-normalization-v1;train-horizontal-flip',
                'optimizer': 'AdamW', 'selection': 'base validation accuracy; loss tie-break',
                'classes': BASE_CLASSES, 'records': groups}
        plan_id = key(plan)
        directory = Path(storage) / 'cache/weights/base-training' / plan_id
        directory.mkdir(parents=True, exist_ok=True)
        completed, best = directory / 'completed.json', directory / 'best.pt'
        if completed.exists():
            artifact = json.loads(completed.read_text())
            if artifact['plan_id'] != plan_id or not best.exists() or digest_file(best) != artifact['sha256']:
                raise ValueError(f'Invalid completed scratch checkpoint: {directory}')
            print(f'CACHE HIT scratch model {name}: {directory}', flush=True)
        else:
            print(f'CACHE MISS scratch model {name}: {directory}', flush=True)
            write_json(directory / 'plan.json', plan)
            metrics = _train(name, source, groups, config, settings, directory, plan_id, device)
            sha256 = digest_file(best)
            artifact = {'plan_id': plan_id, 'initialization': 'random', 'model': name,
                        'architecture_source': MODEL_IDS[name], 'sha256': sha256,
                        'checkpoint_id': f'scratch-base-v1/{name}/{sha256}',
                        'base_train_images': len(groups['train']), 'base_val_images': len(groups['val']),
                        'novel_images_used_for_backbone_training': 0, 'settings': asdict(settings), **metrics}
            write_json(completed, artifact)
        artifacts[name] = {**artifact, 'path': str(best)}
    return artifacts


def load_scratch_backbone(name, device, artifact):
    path = Path(artifact['path'])
    if digest_file(path) != artifact['sha256']:
        raise ValueError(f'Scratch checkpoint checksum mismatch: {path}')
    model = load_backbone(name, device, pretrained=False)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model.requires_grad_(False).eval()
