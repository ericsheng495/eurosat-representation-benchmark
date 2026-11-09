"""Small, inspectable tables and figures for notebook data-processing stages."""
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

from ..config import CLASSES, NOVEL_CLASSES
from ..reporting import STYLE, save_figure
from .images import Images


def episode_table(data, phase='dev', shots=5, episode=0):
    selected = data.episodes[phase][str(shots)][episode]
    return pd.DataFrame([{'phase': phase, 'episode': episode, 'role': role,
                          'class': CLASSES[data.records[i]['label']], 'path': data.records[i]['path']}
                         for role in ['support', 'query'] for i in selected[role]])


def usage_table(data):
    rows = []
    for phase, groups in data.episodes.items():
        for shots, episodes in groups.items():
            for role in ['support', 'query']:
                counts = Counter(i for episode in episodes for i in episode[role])
                rows.append({'phase': phase, 'shots': int(shots), 'role': role,
                             'episodes': len(episodes), 'image_assignments': sum(counts.values()),
                             'unique_images': len(counts), 'maximum_reuse': max(counts.values())})
    result = pd.DataFrame(rows)
    result.to_csv(data.audit_dir / 'episode-usage.csv', index=False)
    return result


def inspect_batch(data, config):
    indices = data.episodes['dev']['1'][0]['support']
    dataset = Images(data.root, data.records, [(i, 0) for i in indices], 'none', config.seed)
    images, labels = next(iter(DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)))
    repeated, _ = next(iter(DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)))
    if tuple(images.shape[1:]) != (3, 224, 224) or not np.isfinite(images.numpy()).all():
        raise ValueError('Invalid model input shape or values')
    if not np.array_equal(images.numpy(), repeated.numpy()):
        raise ValueError('Deterministic development preprocessing changed between reads')
    return {'batch_shape': list(images.shape), 'dtype': str(images.dtype),
            'labels': labels.tolist(), 'finite': True, 'deterministic': True,
            'range_after_normalization': [float(images.min()), float(images.max())]}


def data_figures(data, validated):
    """Use development support images only; never pick examples using query scores."""
    output = data.audit_dir
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 3), layout='constrained')
        frame = pd.DataFrame(validated.pools)
        for pool, color in [('dev_pool', '#376D95'), ('test_pool', '#A7C5D9')]:
            values = frame[frame.pool == pool].set_index('class').loc[NOVEL_CLASSES, 'images']
            bottom = np.zeros(4) if pool == 'dev_pool' else frame[frame.pool == 'dev_pool'].set_index('class').loc[NOVEL_CLASSES, 'images'].to_numpy()
            ax.bar(NOVEL_CLASSES, values, bottom=bottom, color=color, label=pool)
        ax.set(ylabel='Original images', title='Fixed novel-class pools: development and final evaluation')
        ax.tick_params(axis='x', labelsize=8)
        ax.legend(frameon=False)
        save_figure(fig, output, 'novel_pools')

        fig, axes = plt.subplots(1, 4, figsize=(8, 2.5), layout='constrained')
        indices = data.episodes['dev']['1'][0]['support']
        for ax, index in zip(axes, indices):
            row = data.records[index]
            with Image.open(data.root / row['path']) as image:
                ax.imshow(image.convert('RGB'), interpolation='nearest')
            ax.set_title(CLASSES[row['label']], fontsize=8)
            ax.axis('off')
        fig.suptitle('Unaugmented development support images — episode 0, one per class')
        save_figure(fig, output, 'support_examples')

    return {name: str(output / f'{name}.png') for name in ['novel_pools', 'support_examples']}
