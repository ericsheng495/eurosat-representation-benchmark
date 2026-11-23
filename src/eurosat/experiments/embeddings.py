"""Frozen-backbone embeddings, cached on disk so each image and view is computed once per backbone.

Everything a backbone still needs (originals and every augmented view) is computed in a single pass,
so image-loading processes start once per backbone rather than once per set.
"""
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..cache import atomic, key
from ..models import EMBEDDING_DIMENSIONS
from ..preprocessing import Images

PREPROCESS = 'rgb-bilinear224-imagenet-normalization-v1'


def embedding_file(storage, backbone, dataset_id, indices, policy, view, params, seed):
    """Cache path. It changes whenever anything that decides the embeddings changes."""
    identity = {'backbone': backbone, 'dataset': dataset_id, 'images': key(list(map(int, indices))),
                'policy': policy, 'view': view, 'seed': seed, 'preprocess': PREPROCESS}
    if policy != 'none':
        identity['params'] = params
    return Path(storage) / 'cache/embeddings' / f'{key(identity)}.npy'


def save_array(path, array):
    def write(temporary):
        with temporary.open('wb') as stream:
            np.save(stream, array)
    atomic(path, write)


class EmbeddingStore:
    """Embeddings for one backbone. The model is loaded only if something is not cached yet."""

    def __init__(self, storage, backbone_id, model_name, load_model, prepared, config, device):
        self.storage, self.backbone_id, self.model_name = storage, backbone_id, model_name
        self.load_model, self.prepared, self.config, self.device = load_model, prepared, config, device
        self.width = EMBEDDING_DIMENSIONS[model_name]
        self.model, self.stats = None, {'computed_images': 0, 'computed_seconds': 0.0, 'cached_sets': 0, 'computed_sets': 0}

    def get_many(self, requests):
        """requests = [(image indices, policy, view), ...] -> one L2-normalized array per request, in order."""
        arrays, missing = [None] * len(requests), []
        for i, (indices, policy, view) in enumerate(requests):
            path = embedding_file(self.storage, self.backbone_id, self.prepared.dataset_id, indices, policy, view,
                                  self.config.augmentation.get(policy, {}), self.config.seed)
            cached = np.load(path) if path.is_file() else None
            if cached is not None and cached.shape == (len(indices), self.width) and np.isfinite(cached).all():
                arrays[i] = cached
                self.stats['cached_sets'] += 1
            else:
                missing.append((i, path))
        if missing:
            items = [(int(image), requests[i][1], requests[i][2]) for i, _ in missing for image in requests[i][0]]
            features, start = self._extract(items), 0
            for i, path in missing:
                arrays[i] = features[start:start + len(requests[i][0])]
                start += len(requests[i][0])
                save_array(path, arrays[i])
            self.stats['computed_sets'] += len(missing)
        return arrays

    def _extract(self, items):
        if self.model is None:
            started = time.perf_counter()
            self.model = self.load_model()
            self.stats['load_seconds'] = round(time.perf_counter() - started, 2)
        started = time.perf_counter()
        dataset = Images(self.prepared.root, self.prepared.records, items, seed=self.config.seed,
                         augmentation=self.config.augmentation)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False,
                            num_workers=self.config.workers, pin_memory=self.device.type == 'cuda')
        rows = []
        with torch.inference_mode():
            for images, _ in loader:
                rows.append(self.model(images.to(self.device, non_blocking=True)).flatten(1).float().cpu().numpy())
        features = np.concatenate(rows)
        if features.shape != (len(items), self.width) or not np.isfinite(features).all():
            raise ValueError(f'{self.model_name}: expected {len(items)} finite embeddings of width {self.width}')
        self.stats['computed_seconds'] += time.perf_counter() - started
        self.stats['computed_images'] += len(items)
        return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)

    def release(self):
        if self.model is not None:
            del self.model
            self.model = None
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
