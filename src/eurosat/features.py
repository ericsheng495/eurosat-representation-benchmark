"""Chunked original/augmented feature extraction and recovery."""
import time
import zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader
from .preprocessing.images import Images
from .cache import key, write_npz
from .config import MODEL_IDS
from .models import EMBEDDING_DIMENSIONS
from .runtime import environment, memory_usage



def extract(storage, root, records, dataset_id, model_name, model, device, items, policy, config, stats=None):
    identity = {'dataset': dataset_id, 'checkpoint': MODEL_IDS[model_name],
                'preprocess': 'rgb-bilinear224-imagenet-normalization-v1',
                'policy': policy, 'seed': config.seed, 'versions': environment()}
    strength = config.policy_strengths.get(policy, 1.)
    if strength != 1.:
        identity['strength'] = strength
    started = time.perf_counter()
    cached_chunks = computed_chunks = 0
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    matrices, labels = [], []
    for start in range(0, len(items), config.chunk_size):
        chunk_items = items[start:start + config.chunk_size]
        path = storage / 'cache/features' / (key({'config': identity, 'items': chunk_items}) + '.npz')
        features = targets = None
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as cached:
                    features, targets = cached['features'], cached['labels']
                expected = np.array([records[i]['label'] for i, _ in chunk_items])
                if (features.ndim != 2 or features.shape[0] != len(chunk_items) or features.shape[1] < 1
                        or features.shape[1] != EMBEDDING_DIMENSIONS[model_name]
                        or not np.isfinite(features).all() or not np.array_equal(targets, expected)):
                    raise ValueError('Invalid cached feature shape, values or labels')
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
                print(f'CACHE MISS {model_name}/{policy}: invalid chunk {path.name}: {error}', flush=True)
                features = targets = None
        if features is not None:
            cached_chunks += 1
            print(f'CACHE HIT features {model_name}/{policy}: {start}:{start + len(chunk_items)}', flush=True)
        else:
            print(f'CACHE MISS features {model_name}/{policy}: {start}:{start + len(chunk_items)}', flush=True)
            computed_chunks += 1
            dataset = Images(root, records, chunk_items, policy, config.seed, strength)
            loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
            features, targets = [], []
            with torch.inference_mode():
                for images, y in loader:
                    batch = model(images.to(device)).flatten(1).float().cpu().numpy()
                    if batch.shape[1] != EMBEDDING_DIMENSIONS[model_name] or not np.isfinite(batch).all():
                        raise ValueError(f'{model_name}: expected finite embeddings of width {EMBEDDING_DIMENSIONS[model_name]}')
                    features.append(batch)
                    targets.append(y.numpy())
            features, targets = np.concatenate(features), np.concatenate(targets)
            features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
            write_npz(path, features=features, labels=targets)
            print(f'Saved {model_name}/{policy}: {start}:{start + len(chunk_items)}', flush=True)
        matrices.append(features)
        labels.append(targets)
    if not matrices:
        raise ValueError('No images selected for feature extraction')
    if stats is not None:
        stats.update({'model': model_name, 'policy': policy, 'strength': strength,
                      'seconds': time.perf_counter() - started, 'items': len(items),
                      'cached_chunks': cached_chunks, 'computed_chunks': computed_chunks,
                      **memory_usage(device)})
    return np.concatenate(matrices), np.concatenate(labels)
