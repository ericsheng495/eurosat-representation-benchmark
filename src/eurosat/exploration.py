"""Dataset audit and reproducible figures for the notebook and project report."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import csv
import io
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from .augment import image_tensor
from .cache import atomic, digest_file, key, prepare_storage, write_json
from .config import CLASSES, Config
from .data import prepare_data


CLASS_LABELS = ['Annual crop', 'Forest', 'Herbaceous vegetation', 'Highway', 'Industrial',
                'Pasture', 'Permanent crop', 'Residential', 'River', 'Sea / lake']
FIGURE_NAMES = ['class_distribution', 'example_tile', 'class_examples',
                'cleaning_audit', 'stratified_shuffle', 'augmentation_examples']
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 9,
         'axes.labelsize': 9, 'axes.spines.top': False, 'axes.spines.right': False,
         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'}


def dataset_profile(root, records, split, duplicates_removed=0):
    """Measure native image properties and verify partition isolation."""
    sizes, modes, file_bytes = Counter(), Counter(), []
    for record in records:
        path = root / record['path']
        with Image.open(path) as image:
            sizes[f'{image.width}x{image.height}'] += 1
            modes[image.mode] += 1
        file_bytes.append(path.stat().st_size)
    members = [set(indices) for indices in split.values()]
    overlap = set()
    pixel_overlap = set()
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            overlap |= members[i] & members[j]
            pixel_overlap |= ({records[k]['pixels_sha256'] for k in members[i]} &
                              {records[k]['pixels_sha256'] for k in members[j]})
    if overlap or pixel_overlap:
        raise ValueError('Dataset partitions overlap by image or decoded pixels')
    rows = []
    for label, name in enumerate(CLASSES):
        row = {'class': name, 'label': CLASS_LABELS[label],
               'images': sum(r['label'] == label for r in records)}
        row.update({part: sum(records[i]['label'] == label for i in indices)
                    for part, indices in split.items()})
        rows.append(row)
    counts = [r['images'] for r in rows]
    return {'images': len(records), 'classes': len(CLASSES), 'native_sizes': dict(sizes),
            'color_modes': dict(modes), 'duplicates_removed': duplicates_removed,
            'original_images': len(records) + duplicates_removed, 'corrupt_images': 0,
            'validation': 'Archive checksum and decoded RGB pixels checked before partitioning',
            'partition_counts': {name: len(ids) for name, ids in split.items()},
            'partition_overlap_images': len(overlap), 'partition_overlap_pixels': len(pixel_overlap),
            'largest_to_smallest_class_ratio': max(counts) / min(counts),
            'image_file_bytes': {'total': sum(file_bytes), 'min': min(file_bytes),
                                 'median': float(np.median(file_bytes)), 'max': max(file_bytes)},
            'class_counts': rows,
            'source': 'EuroSAT RGB, Helber et al. (2019), Copernicus Sentinel-2',
            'source_url': 'https://github.com/phelber/EuroSAT'}


def dataset_figures(root, records, split, profile, output, seed=42):
    """Save 300-DPI PNG and vector PDF figures; examples come only from support."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    examples = [rng.choice([i for i in split['support'] if records[i]['label'] == label])
                for label in range(len(CLASSES))]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 4.1), layout='constrained')
        y = np.arange(len(CLASSES))
        offset = np.zeros(len(CLASSES))
        for name, title, color in [('support', 'Support pool', '#376D95'),
                                   ('dev', 'Development', '#6EA6A1'), ('test', 'Held-out test', '#CFD5DC')]:
            values = np.array([row[name] for row in profile['class_counts']])
            ax.barh(y, values, left=offset, height=.68, label=title, color=color)
            offset += values
        for i, total in enumerate(offset):
            ax.text(total + max(offset) * .015, i, f'{int(total):,}', va='center', fontsize=8)
        ax.set_yticks(y, CLASS_LABELS)
        ax.invert_yaxis()
        ax.set_xlim(0, max(offset) * 1.16)
        ax.set_xlabel('Number of original images')
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        ax.legend(ncol=3, frameon=False, loc='lower left', bbox_to_anchor=(0, 1.01), borderaxespad=0)
        save_figure(fig, output, 'class_distribution')

        example = records[examples[0]]
        with Image.open(root / example['path']) as image:
            native = np.asarray(image.convert('RGB'))
        fig, ax = plt.subplots(figsize=(3.5, 3.5), layout='constrained')
        ax.imshow(native, interpolation='nearest')
        ax.set_title(CLASS_LABELS[0], loc='left', fontweight='bold')
        ax.axis('off')
        height = native.shape[0]
        ax.plot([4, 14], [height - 6, height - 6], color='white', linewidth=2.5)
        ax.text(9, height - 8, '10 px', color='white', ha='center', va='bottom', fontsize=8)
        save_figure(fig, output, 'example_tile')

        fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.5), layout='constrained')
        for label, (ax, image_id) in enumerate(zip(axes.flat, examples)):
            with Image.open(root / records[image_id]['path']) as image:
                ax.imshow(image.convert('RGB'), interpolation='nearest')
            ax.set_title(f'({chr(97 + label)}) {CLASS_LABELS[label]}', loc='left', fontsize=7.5)
            ax.axis('off')
        save_figure(fig, output, 'class_examples')

        cleaning = pd.DataFrame({'stage': ['Original archive', 'After duplicate removal'],
                                 'images': [profile['original_images'], profile['images']]})
        fig, ax = plt.subplots(figsize=(5.2, 2.6), layout='constrained')
        sns.barplot(data=cleaning, x='images', y='stage', color='#376D95', ax=ax)
        for i, value in enumerate(cleaning['images']):
            ax.text(value + max(cleaning['images']) * .02, i, f'{value:,}', va='center')
        ax.set_xlim(0, max(cleaning['images']) * 1.2)
        ax.set(xlabel='Original RGB images', ylabel=None)
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        save_figure(fig, output, 'cleaning_audit')

        # This is the actual first-class permutation used to construct the partitions.
        ordered = [i for i, row in enumerate(records) if row['label'] == 0]
        shuffled = [i for phase in ['support', 'dev', 'test'] for i in split[phase]
                    if records[i]['label'] == 0]
        rank = {image_id: i for i, image_id in enumerate(ordered)}
        positions = np.linspace(0, len(ordered) - 1, min(300, len(ordered)), dtype=int)
        support_end = sum(records[i]['label'] == 0 for i in split['support'])
        dev_end = support_end + sum(records[i]['label'] == 0 for i in split['dev'])
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), layout='constrained', sharey=True)
        for ax, order, title in zip(axes, [ordered, shuffled], ['Before shuffling', 'After stratified shuffling']):
            frame = pd.DataFrame({'Position': positions, 'Original index': [rank[order[i]] for i in positions]})
            sns.scatterplot(data=frame, x='Position', y='Original index', ax=ax,
                            s=9, color='#376D95', alpha=.7, linewidth=0)
            ax.set_title(title, loc='left')
        for boundary in [support_end, dev_end]:
            axes[1].axvline(boundary, linestyle='--', color='#687682', linewidth=.7)
        axes[0].set_ylabel('Original within-class index')
        save_figure(fig, output, 'stratified_shuffle')

        policies = ['none', 'none', 'flip', 'rotation', 'translation', 'scale',
                    'brightness_contrast', 'noise', 'blur', 'erasing']
        titles = ['Original', 'Repeated original', 'Flip', 'Rotation', 'Translation', 'Scale',
                  'Brightness / contrast', 'Noise', 'Blur', 'Erasing']
        fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.5), layout='constrained')
        mean = np.array([.485, .456, .406])
        std = np.array([.229, .224, .225])
        for ax, policy, title in zip(axes.flat, policies, titles):
            view_seed = int(key([seed, example['pixels_sha256'], policy, 0])[:15], 16)
            tensor = image_tensor(root / example['path'], policy, view_seed)
            rgb = np.clip(tensor.numpy().transpose(1, 2, 0) * std + mean, 0, 1)
            ax.imshow(rgb, interpolation='nearest')
            ax.set_title(title, loc='left', fontsize=7.5)
            ax.axis('off')
        save_figure(fig, output, 'augmentation_examples')
    write_json(output / 'figure_sources.json',
               {'seed': seed, 'partition': 'support', 'example': example['path'],
                'class_examples': [records[i]['path'] for i in examples],
                'native_pixels': True, 'native_example_augmentation': 'none',
                'augmentation_montage_view': 0, 'geographic_scale': 'not inferred'})
    return {name: {ext: str(output / f'{name}.{ext}') for ext in ['png', 'pdf']}
            for name in FIGURE_NAMES}


def save_figure(figure, output, name):
    for extension in ['png', 'pdf']:
        figure.savefig(output / f'{name}.{extension}', dpi=300, bbox_inches='tight', pad_inches=.04)
    plt.close(figure)


def explore(storage, scratch, config=None, output=None):
    """Profile the entire dataset even when experiment execution uses a smoke subset."""
    config = replace(config or Config(), per_class_limit=None)
    storage = prepare_storage(storage)
    root, records, split, dataset_id = prepare_data(storage, scratch, config)
    manifest_path = storage / 'cache' / ('dataset-' + digest_file(storage / 'data/EuroSAT.zip') + '.json')
    duplicates = len(json.loads(manifest_path.read_text())['duplicates'])
    profile = dataset_profile(root, records, split, duplicates)
    profile['dataset_id'] = dataset_id
    profile['scope'] = 'Full deduplicated dataset; experiment smoke limits are not applied here'
    output = Path(output) if output else storage / 'outputs/dataset' / dataset_id
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'dataset_profile.json', profile)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, lineterminator='\n', fieldnames=['class', 'label', 'images', 'support', 'dev', 'test'])
    writer.writeheader()
    writer.writerows(profile['class_counts'])
    atomic(output / 'class_counts.csv', lambda p: p.write_text(stream.getvalue()))
    figures = dataset_figures(root, records, split, profile, output, config.seed)
    print(f'Dataset profile: {profile["images"]:,} images, {profile["classes"]} classes, '
          f'native sizes {profile["native_sizes"]}; figures: {output}', flush=True)
    return {'profile': profile, 'figures': figures, 'output': str(output)}
