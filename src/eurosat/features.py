"""Chunked original/augmented feature extraction and recovery."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .augment import image_tensor
from .cache import key, write_npz
from .config import MODEL_IDS
from .runtime import environment


class Images(Dataset):
    def __init__(self, root, records, items, policy, seed):
        self.root, self.records, self.items = root, records, items
        self.policy, self.seed = policy, seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_id, view = self.items[index]
        record = self.records[image_id]
        # Image/view seeds are shared across backbones and independent of batch/chunk order.
        seed = int(key([self.seed, record['pixels_sha256'], self.policy, view])[:15], 16)
        return image_tensor(self.root / record['path'], self.policy, seed), record['label']


def extract(storage, root, records, dataset_id, model_name, model, device, items, policy, config):
    identity = {'dataset': dataset_id, 'checkpoint': MODEL_IDS[model_name],
                'preprocess': 'rgb-bilinear224-imagenet-normalization-v1',
                'policy': policy, 'seed': config.seed, 'versions': environment()}
    matrices, labels = [], []
    for start in range(0, len(items), config.chunk_size):
        chunk_items = items[start:start + config.chunk_size]
        path = storage / 'cache/features' / (key({'config': identity, 'items': chunk_items}) + '.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                features, targets = cached['features'], cached['labels']
            print(f'Cached {model_name}/{policy}: {start}:{start + len(chunk_items)}', flush=True)
        else:
            dataset = Images(root, records, chunk_items, policy, config.seed)
            loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
            features, targets = [], []
            with torch.inference_mode():
                for images, y in loader:
                    features.append(model(images.to(device)).flatten(1).float().cpu().numpy())
                    targets.append(y.numpy())
            features, targets = np.concatenate(features), np.concatenate(targets)
            features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
            write_npz(path, features=features, labels=targets)
            print(f'Saved {model_name}/{policy}: {start}:{start + len(chunk_items)}', flush=True)
        matrices.append(features)
        labels.append(targets)
    if not matrices:
        raise ValueError('No images selected for feature extraction')
    return np.concatenate(matrices), np.concatenate(labels)
