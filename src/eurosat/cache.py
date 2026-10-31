"""Content-addressed artifacts and atomic checkpoint publication."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import numpy as np
from .config import PIPELINE_VERSION


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


def prepare_storage(storage):
    import torch
    storage = Path(storage)
    for name in ['data', 'cache/weights', 'cache/features', 'cache/experiments', 'outputs']:
        (storage / name).mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(storage / 'cache/weights/torch'))
    return storage
