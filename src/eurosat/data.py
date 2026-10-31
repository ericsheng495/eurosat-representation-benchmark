"""Verified EuroSAT acquisition, duplicate checking, partitions and paired trials."""
from pathlib import Path
import hashlib
import json
import random
import shutil
import urllib.request
import zipfile
from PIL import Image
from .cache import atomic, digest_file, key, write_json
from .config import CLASSES, DATA_URL, DATA_MD5


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
