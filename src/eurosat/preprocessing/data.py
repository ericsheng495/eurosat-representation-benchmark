"""Verified archive acquisition and a reusable audit of the extracted image files."""
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import hashlib
import json
import random
import shutil
import urllib.request
import zipfile
import zlib

from PIL import Image

from ..cache import atomic, digest_file, key, prepare_storage, write_json
from ..config import BASE_CLASSES, CLASSES, DATA_URL, DATA_MD5


@dataclass
class SourceDataset:
    root: Path
    records: list
    archive_sha256: str
    source_id: str
    inventory: list
    properties: dict
    integrity: dict
    audit_path: Path


def _source_cache_id(records, seed):
    """Retain existing feature identifiers; these legacy hash fields do not assign episodes."""
    rng = random.Random(seed)
    fields = {'support': [], 'dev': [], 'test': []}

    for label in range(len(CLASSES)):
        indices = [i for i, row in enumerate(records) if row['label'] == label]
        rng.shuffle(indices)
        first, second = int(len(indices) * .6), int(len(indices) * .2)
        fields['support'].extend(indices[:first])
        fields['dev'].extend(indices[first:first + second])
        fields['test'].extend(indices[first + second:])

    return key({'records': records, 'split': fields})


def load_source(storage, scratch, seed=42):
    """Verify the ZIP and extracted bytes; reuse the decoded-pixel audit when valid."""
    storage = prepare_storage(storage)
    archive = storage / 'data/EuroSAT.zip'

    if not archive.exists():
        def download(temporary):
            print('CACHE MISS archive: downloading EuroSAT RGB', flush=True)
            request = urllib.request.Request(DATA_URL, headers={'User-Agent': 'eurosat-representation-benchmark/0.1'})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as stream:
                shutil.copyfileobj(response, stream)
            if digest_file(temporary, 'md5') != DATA_MD5:
                raise ValueError('Downloaded archive checksum mismatch')
        atomic(archive, download)

    if digest_file(archive, 'md5') != DATA_MD5:
        raise ValueError(f'Archive checksum mismatch: {archive}')

    archive_id = digest_file(archive)
    raw = Path(scratch) / archive_id[:16]
    raw.mkdir(parents=True, exist_ok=True)
    repaired = 0

    with zipfile.ZipFile(archive) as bundle:
        images = [info for info in bundle.infolist() if not info.is_dir()
                  and Path(info.filename).suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}]
        for info in images:
            target = raw / info.filename
            if not target.resolve().is_relative_to(raw.resolve()):
                raise ValueError('Archive contains an unsafe image path')
            if (not target.is_file() or target.stat().st_size != info.file_size
                    or zlib.crc32(target.read_bytes()) != info.CRC):
                target.parent.mkdir(parents=True, exist_ok=True)
                atomic(target, lambda p, info=info: p.write_bytes(bundle.read(info)))
                repaired += 1

    (raw / '.complete').write_text(archive_id)
    candidates = [raw, *sorted(p for p in raw.rglob('*') if p.is_dir())]
    root = next((p for p in candidates if all((p / name).is_dir() for name in CLASSES)), None)
    if root is None:
        raise ValueError('The archive does not contain the ten expected class folders')

    audit_path = storage / 'cache' / f'dataset-{archive_id}.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    if audit is not None and audit.get('archive_sha256') != archive_id:
        raise ValueError('Cached image audit belongs to a different archive')

    if audit is None or 'properties' not in audit:
        records, duplicates, seen = [], [], {}
        sizes, modes = Counter(), Counter()

        for label, name in enumerate(CLASSES):
            for path in sorted((root / name).glob('*.jpg')):
                try:
                    with Image.open(path) as image:
                        image.load()
                        sizes[f'{image.width}x{image.height}'] += 1
                        modes[image.mode] += 1
                        rgb = image.convert('RGB')
                        fingerprint = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                except (OSError, ValueError) as error:
                    raise ValueError(f'Cannot decode image {path.relative_to(root)}: {error}') from error

                relative = path.relative_to(root).as_posix()
                if fingerprint in seen:
                    previous = seen[fingerprint]
                    if previous['label'] != label:
                        raise ValueError(f'Identical pixels have conflicting labels: {relative}')
                    duplicates.append({'kept': previous['path'], 'removed': relative})
                    continue

                row = {'path': relative, 'label': label, 'pixels_sha256': fingerprint}
                seen[fingerprint] = row
                records.append(row)

        audit = {'archive_sha256': archive_id, 'records': records, 'duplicates': duplicates,
                 'properties': {'native_sizes': dict(sizes), 'color_modes': dict(modes)}}
        write_json(audit_path, audit)

    records = audit['records']
    inventory = [{'class': name, 'role': 'base (reserved)' if name in BASE_CLASSES else 'novel (evaluated)',
                  'archive_images': sum(Path(info.filename).parent.name == name for info in images),
                  'unique_images': sum(row['label'] == label for row in records)}
                 for label, name in enumerate(CLASSES)]
    integrity = {'archive_md5_verified': True, 'files_crc_verified': len(images),
                 'missing_or_changed_files_restored': repaired,
                 'decoded_pixel_duplicates': len(audit['duplicates']), 'corrupt_images': 0}

    print(f'Source audit: {len(images):,} files verified; {repaired} restored; '
          f'{len(audit["duplicates"])} exact-pixel duplicates', flush=True)

    return SourceDataset(root, records, archive_id, _source_cache_id(records, seed),
                         inventory, audit['properties'], integrity, audit_path)
