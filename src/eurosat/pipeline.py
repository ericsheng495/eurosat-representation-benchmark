"""Portable pipeline; also embedded in Start_Here.ipynb for single-file Colab use."""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import io
import json
import os
import random
import shutil
import tempfile
import time
import urllib.request
import zipfile

import numpy as np
from PIL import Image
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import models, transforms
from torchvision.transforms import functional as TF, InterpolationMode


PIPELINE_VERSION = '1'
DATA_URL = ('https://huggingface.co/datasets/torchgeo/eurosat/resolve/'
            'c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip')
DATA_MD5 = 'c8fa014336c82ac7804f0398fcb19387'
DINO_REVISION = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
MODEL_IDS = {
    'resnet18': 'torchvision.ResNet18_Weights.IMAGENET1K_V1',
    'convnext_tiny': 'torchvision.ConvNeXt_Tiny_Weights.IMAGENET1K_V1',
    'vit_b_16': 'torchvision.ViT_B_16_Weights.IMAGENET1K_V1',
    'dinov2_vits14': 'facebookresearch/dinov2:' + DINO_REVISION + '/dinov2_vits14',
}
POLICIES = ['none', 'repeat', 'flip', 'rotation', 'translation', 'scale',
            'brightness_contrast', 'noise', 'blur', 'erasing']


@dataclass
class Config:
    mode: str = 'smoke'
    seed: int = 42
    models: list = field(default_factory=lambda: ['resnet18'])
    policies: list = field(default_factory=lambda: ['none', 'repeat', 'flip'])
    shots: list = field(default_factory=lambda: [1, 5, 10])
    trials: int = 1
    views: int = 1
    per_class_limit: int | None = 40
    batch_size: int = 32
    chunk_size: int = 256
    classifier_cs: list = field(default_factory=lambda: [0.1, 1.0, 10.0])

    @classmethod
    def full(cls):
        return cls(mode='full', models=list(MODEL_IDS), policies=POLICIES.copy(),
                   trials=20, views=4, per_class_limit=None)

    def validate(self):
        if self.mode not in {'smoke', 'full'}:
            raise ValueError('mode must be smoke or full')
        if not self.models or set(self.models) - MODEL_IDS.keys():
            raise ValueError('Choose supported models: ' + ', '.join(MODEL_IDS))
        if 'none' not in self.policies or set(self.policies) - set(POLICIES):
            raise ValueError('Policies must include none and contain supported names')
        if not self.shots or min(self.shots) < 1 or min(self.trials, self.views, self.batch_size, self.chunk_size) < 1:
            raise ValueError('Shot budgets, trials, views and batch/chunk sizes must be positive')
        if not self.classifier_cs or min(self.classifier_cs) <= 0:
            raise ValueError('Classifier C values must be positive')


def key(value):
    text = json.dumps({'pipeline': PIPELINE_VERSION, 'value': value},
                      sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def digest_file(path, algorithm='sha256'):
    result = hashlib.new(algorithm)
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def atomic(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    os.close(fd)
    try:
        write(Path(temporary))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, value):
    atomic(path, lambda p: p.write_text(json.dumps(value, indent=2, sort_keys=True,
                                                   allow_nan=False) + '\n'))


def write_npz(path, **arrays):
    def save(temporary):
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
    atomic(path, save)


def environment():
    return {'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
            'numpy': np.__version__, 'sklearn': sklearn.__version__}


def check_device(target='colab_gpu'):
    if target not in {'colab_gpu', 'local'}:
        raise ValueError('target must be colab_gpu or local')
    if target == 'colab_gpu':
        if not Path('/content').is_dir() or not torch.cuda.is_available():
            raise RuntimeError('Select a Colab GPU server (T4 or L4), not a local/CPU/TPU '
                               'kernel. In VS Code: Select Another Kernel > Colab > '
                               'New Colab Server > GPU > Python 3 (ipykernel).')
        device = torch.device('cuda')
        label = torch.cuda.get_device_name(0)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else
                              'mps' if torch.backends.mps.is_available() else 'cpu')
        label = str(device)
    if device.type == 'cpu':
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    if (torch.ones(1, device=device) + 1).cpu().item() != 2:
        raise RuntimeError('Device operation check failed')
    print(f'Compute: {label} [{device}] — device check passed', flush=True)
    return device


def prepare_storage(storage):
    storage = Path(storage)
    for name in ['data', 'cache/weights', 'cache/features', 'cache/experiments', 'outputs']:
        (storage / name).mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(storage / 'cache/weights/torch'))
    return storage


def prepare_data(storage, scratch, config):
    """Persist one archive in Drive; read extracted images from fast runtime-local disk."""
    archive = storage / 'data/EuroSAT.zip'
    if not archive.exists():
        def download(temporary):
            print('Downloading EuroSAT RGB (~90 MB), once per storage folder...', flush=True)
            request = urllib.request.Request(DATA_URL, headers={'User-Agent': 'eurosat-low-data/0.1'})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as stream:
                shutil.copyfileobj(response, stream)
            if digest_file(temporary, 'md5') != DATA_MD5:
                raise ValueError('Downloaded EuroSAT checksum does not match; rerun to retry')
        atomic(archive, download)
    if digest_file(archive, 'md5') != DATA_MD5:
        raise ValueError(f'EuroSAT archive is corrupt: {archive}')
    archive_id = digest_file(archive)
    raw = Path(scratch) / archive_id[:16]
    marker = raw / '.complete'
    if not marker.exists():
        raw.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            for name in bundle.namelist():
                if not (raw / name).resolve().is_relative_to(raw.resolve()):
                    raise ValueError('Archive contains an invalid path')
            bundle.extractall(raw)
        marker.write_text(archive_id)
    candidates = [raw] + sorted(p for p in raw.rglob('*') if p.is_dir())
    image_root = next((p for p in candidates if all((p / name).is_dir() for name in CLASSES)), None)
    if image_root is None:
        raise ValueError('EuroSAT class folders not found')
    manifest_path = storage / 'cache' / ('dataset-' + archive_id + '.json')
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        records, duplicates, seen = [], [], {}
        for label, name in enumerate(CLASSES):
            for path in sorted((image_root / name).glob('*.jpg')):
                with Image.open(path) as image:
                    rgb = image.convert('RGB')
                    fingerprint = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                relative = path.relative_to(image_root).as_posix()
                if fingerprint in seen:
                    previous = seen[fingerprint]
                    if previous['label'] != label:
                        raise ValueError('Identical pixels have conflicting class labels')
                    duplicates.append({'kept': previous['path'], 'removed': relative})
                    continue
                record = {'path': relative, 'label': label, 'pixels_sha256': fingerprint}
                seen[fingerprint] = record
                records.append(record)
        manifest = {'archive_sha256': archive_id, 'records': records, 'duplicates': duplicates}
        write_json(manifest_path, manifest)
    records = manifest['records']
    if config.per_class_limit is not None:
        rng = random.Random(config.seed)
        records = [r for label in range(len(CLASSES)) for r in rng.sample(
            [r for r in records if r['label'] == label],
            min(config.per_class_limit, sum(r['label'] == label for r in records)))]
        records.sort(key=lambda r: r['path'])
    split = stratified_split(records, config.seed)
    dataset_id = key({'records': records, 'split': split})
    write_json(storage / 'cache' / ('split-' + dataset_id + '.json'),
               {'classes': CLASSES, 'records': records, 'split': split,
                'duplicate_images_removed': len(manifest['duplicates'])})
    counts = {name: len(indices) for name, indices in split.items()}
    print(f'Data: {len(records)} unique images, splits {counts}; '
          f'{len(manifest["duplicates"])} exact-pixel duplicates removed', flush=True)
    return image_root, records, split, dataset_id


def stratified_split(records, seed):
    split = {'support': [], 'dev': [], 'test': []}
    rng = random.Random(seed)
    for label in range(len(CLASSES)):
        indices = [i for i, row in enumerate(records) if row['label'] == label]
        rng.shuffle(indices)
        train, dev = int(len(indices) * .6), int(len(indices) * .2)
        if min(train, dev, len(indices) - train - dev) < 1:
            raise ValueError('Each class needs enough distinct images for three partitions')
        split['support'].extend(indices[:train])
        split['dev'].extend(indices[train:train + dev])
        split['test'].extend(indices[train + dev:])
    return split


def make_trials(records, split, config):
    trials = []
    maximum = max(config.shots)
    for trial in range(config.trials):
        rng = random.Random(config.seed + trial)
        selected = {}
        for label in range(len(CLASSES)):
            pool = [i for i in split['support'] if records[i]['label'] == label]
            if len(pool) < maximum:
                raise ValueError(f'Class {CLASSES[label]} has fewer than {maximum} support-pool images')
            # Shuffle the whole pool so changing the requested shot/view grid cannot
            # change the original support images in a corresponding trial.
            rng.shuffle(pool)
            selected[label] = pool[:maximum]
        trials.append({str(k): [i for label in range(len(CLASSES)) for i in selected[label][:k]]
                       for k in sorted(config.shots)})
    return trials


def load_backbone(name, device):
    if name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
    elif name == 'convnext_tiny':
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Identity()
    elif name == 'vit_b_16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = nn.Identity()
    elif name == 'dinov2_vits14':
        # Official implementation + weights pinned to a specific source revision.
        os.environ['XFORMERS_DISABLED'] = '1'
        model = torch.hub.load('facebookresearch/dinov2:' + DINO_REVISION,
                               'dinov2_vits14', pretrained=True,
                               trust_repo=True, skip_validation=True)
    else:
        raise ValueError('Unknown backbone: ' + name)
    model.requires_grad_(False)
    return model.to(device).eval()


def image_tensor(path, policy, seed):
    rng = random.Random(seed)
    with Image.open(path) as image:
        image = TF.resize(image.convert('RGB'), [224, 224], InterpolationMode.BILINEAR)
    if policy == 'flip':
        image = TF.hflip(image) if rng.random() < .5 else TF.vflip(image)
    elif policy == 'rotation':
        image = TF.rotate(image, rng.uniform(-30, 30), InterpolationMode.BILINEAR)
    elif policy == 'translation':
        image = TF.affine(image, 0, [rng.randint(-22, 22), rng.randint(-22, 22)], 1., [0., 0.],
                          InterpolationMode.BILINEAR)
    elif policy == 'scale':
        image = TF.affine(image, 0, [0, 0], rng.uniform(.8, 1.2), [0., 0.], InterpolationMode.BILINEAR)
    elif policy == 'brightness_contrast':
        image = TF.adjust_contrast(TF.adjust_brightness(image, rng.uniform(.7, 1.3)), rng.uniform(.7, 1.3))
    elif policy == 'blur':
        image = TF.gaussian_blur(image, [5, 5], [rng.uniform(.1, 1.5)] * 2)
    elif policy not in {'none', 'noise', 'erasing', 'cutout'}:
        raise ValueError('Unsupported image augmentation: ' + policy)
    tensor = TF.to_tensor(image)
    if policy == 'noise':
        generator = torch.Generator().manual_seed(seed)
        tensor = (tensor + torch.randn(tensor.shape, generator=generator) * .05).clamp(0, 1)
    elif policy in {'erasing', 'cutout'}:
        size = rng.randint(22, 67)
        top, left = rng.randrange(225 - size), rng.randrange(225 - size)
        tensor[:, top:top + size, left:left + size] = .5 if policy == 'erasing' else 0.
    return TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


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


def score(labels, predictions):
    labels_all = list(range(len(CLASSES)))
    return {'accuracy': float(accuracy_score(labels, predictions)),
            'macro_f1': float(f1_score(labels, predictions, labels=labels_all, average='macro', zero_division=0)),
            'class_recall': recall_score(labels, predictions, labels=labels_all,
                                         average=None, zero_division=0).tolist(),
            'confusion_matrix': confusion_matrix(labels, predictions, labels=labels_all).tolist()}


def fit_classifier(x, y, dev_x, dev_y, config):
    best, selected, dev_accuracy = None, None, -1.
    for c in sorted(config.classifier_cs):
        # lbfgs uses multinomial loss for this ten-class task and L2 regularization.
        classifier = LogisticRegression(C=c, solver='lbfgs', max_iter=2000,
                                        random_state=config.seed)
        classifier.fit(x, y)
        accuracy = classifier.score(dev_x, dev_y)
        if accuracy > dev_accuracy:
            best, selected, dev_accuracy = classifier, c, float(accuracy)
    return best, selected, dev_accuracy


def run(config, storage, scratch, target='colab_gpu'):
    config.validate()
    device = check_device(target)
    storage = prepare_storage(storage)
    root, records, split, dataset_id = prepare_data(storage, scratch, config)
    trials = make_trials(records, split, config)
    manifest = {'config': asdict(config), 'dataset': dataset_id, 'versions': environment(),
                'trials': trials, 'checkpoints': {m: MODEL_IDS[m] for m in config.models}}
    experiment = key(manifest)
    directory = storage / 'cache/experiments' / experiment
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'manifest.json', manifest)
    hardware = (torch.cuda.get_device_name(0) if device.type == 'cuda'
                else 'Apple GPU (MPS)' if device.type == 'mps' else 'CPU')
    write_json(directory / f'execution-{time.time_ns()}.json',
               {'utc': datetime.now(timezone.utc).isoformat(), 'device': str(device),
                'hardware': hardware, 'target': target, 'versions': environment()})
    labels_all = np.array([r['label'] for r in records])
    all_items = [(i, 0) for i in range(len(records))]
    support_union = sorted({i for trial in trials for indices in trial.values() for i in indices})
    rows = []
    for model_name in config.models:
        # A fully completed model is skipped without loading its weights or features.
        result_paths = [directory / (key([model_name, p, trial, k]) + '.json')
                        for p in config.policies for trial in range(config.trials) for k in config.shots]
        if all(p.exists() for p in result_paths):
            rows.extend(json.loads(p.read_text()) for p in result_paths)
            print('Completed results reused:', model_name, flush=True)
            continue
        started = time.perf_counter()
        model = load_backbone(model_name, device)
        original, _ = extract(storage, root, records, dataset_id, model_name, model,
                               device, all_items, 'none', config)
        feature_seconds = time.perf_counter() - started
        for policy in config.policies:
            paths = [directory / (key([model_name, policy, trial, k]) + '.json')
                     for trial in range(config.trials) for k in config.shots]
            if all(path.exists() for path in paths):
                rows.extend(json.loads(path.read_text()) for path in paths)
                continue
            augmented = None
            aug_map = {}
            if policy not in {'none', 'repeat'}:
                items = [(i, v) for i in support_union for v in range(config.views)]
                augmented, _ = extract(storage, root, records, dataset_id, model_name, model,
                                       device, items, policy, config)
                aug_map = {item: j for j, item in enumerate(items)}
            for trial_number, trial in enumerate(trials):
                for k in config.shots:
                    result_path = directory / (key([model_name, policy, trial_number, k]) + '.json')
                    if result_path.exists():
                        rows.append(json.loads(result_path.read_text()))
                        continue
                    indices = trial[str(k)]
                    x, y = original[indices], labels_all[indices]
                    if policy == 'repeat':
                        x, y = np.tile(x, (config.views + 1, 1)), np.tile(y, config.views + 1)
                    elif augmented is not None:
                        aug_indices = [aug_map[i, view] for i in indices for view in range(config.views)]
                        x = np.concatenate([x, augmented[aug_indices]])
                        y = np.concatenate([y, np.repeat(y, config.views)])
                    fit_started = time.perf_counter()
                    classifier, c, dev_accuracy = fit_classifier(
                        x, y, original[split['dev']], labels_all[split['dev']], config)
                    fit_seconds = time.perf_counter() - fit_started
                    result = {'model': model_name, 'policy': policy, 'shots': k, 'trial': trial_number,
                              'original_support_images': len(indices), 'classifier_rows': len(x),
                              'views': 0 if policy == 'none' else config.views, 'C': c,
                              'dev_accuracy': dev_accuracy,
                              'test': score(labels_all[split['test']], classifier.predict(original[split['test']])),
                              'classifier_fit_seconds': fit_seconds, 'original_feature_seconds': feature_seconds,
                              'parameter_count': sum(p.numel() for p in model.parameters())}
                    # Numeric classifier checkpoint: reloadable without pickle/code execution.
                    write_npz(result_path.with_suffix('.npz'), coef=classifier.coef_,
                              intercept=classifier.intercept_, classes=classifier.classes_)
                    write_json(result_path, result)
                    rows.append(result)
                    print(f'{model_name} {policy} {k}-shot trial {trial_number}: '
                          f'{result["test"]["accuracy"]:.3f}', flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()
    summary = summarize(rows)
    output = storage / 'outputs' / experiment
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'results.json', {'manifest': manifest, 'trials': rows, 'summary': summary})
    columns = ['model', 'policy', 'shots', 'trials', 'accuracy', 'accuracy_std', 'macro_f1',
               'paired_gain_percentage_points', 'dev_accuracy']
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(summary)
    atomic(output / 'summary.csv', lambda p: p.write_text(stream.getvalue()))
    # Augmentation selection uses development scores only, with a fixed name tie-break.
    selected = []
    for model_name in config.models:
        for k in config.shots:
            candidates = [r for r in summary if r['model'] == model_name and r['shots'] == k]
            winner = sorted(candidates, key=lambda r: (-r['dev_accuracy'], r['policy']))[0]
            selected.append(winner)
    write_json(output / 'development-selected-policies.json', selected)
    print('Finished. Results:', output, flush=True)
    return {'output': str(output), 'summary': summary, 'experiment': experiment}


def summarize(rows):
    baseline = {(r['model'], r['shots'], r['trial']): r['test']['accuracy']
                for r in rows if r['policy'] == 'none'}
    groups = sorted({(r['model'], r['policy'], r['shots']) for r in rows})
    summary = []
    for model, policy, shots in groups:
        group = [r for r in rows if (r['model'], r['policy'], r['shots']) == (model, policy, shots)]
        scores = [r['test']['accuracy'] for r in group]
        gains = [r['test']['accuracy'] - baseline[model, shots, r['trial']] for r in group]
        summary.append({'model': model, 'policy': policy, 'shots': shots, 'trials': len(group),
                        'accuracy': float(np.mean(scores)), 'accuracy_std': float(np.std(scores)),
                        'macro_f1': float(np.mean([r['test']['macro_f1'] for r in group])),
                        'paired_gain_percentage_points': float(np.mean(gains) * 100),
                        'dev_accuracy': float(np.mean([r['dev_accuracy'] for r in group]))})
    return summary
