"""Configure and run the shared four-class manifest experiment."""
from dataclasses import replace
import json
from pathlib import Path


from .cache import write_json
from .config import Config, MODEL_IDS
from .pipeline import run
from .reporting import create_report


def manifest_config(mode='smoke', models='all'):
    """One four-way protocol; model selection never changes the episode assignments."""
    if mode not in {'full', 'smoke'}:
        raise ValueError('mode must be full or smoke')
    names = models.split(',') if isinstance(models, str) else list(models)
    names = [name.strip() for name in names]
    config = Config.full() if mode == 'full' else Config()
    config.models = list(MODEL_IDS) if names == ['all'] else names
    config.validate()
    return config


def run_manifest_benchmark(config, storage, scratch, manifest_path, target='colab_gpu', episode_limit=None, prepared=None):
    """Use committed development/test episodes with frozen backbones and a common head."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f'Copy the committed splits_manifest.json to shared storage: {path}')
    saved = json.loads(path.read_text())
    available = min(len(saved['episodes']['test'][f'{k}shot']) for k in config.shots)
    count = episode_limit if episode_limit is not None else (min(2, available) if config.mode == 'smoke' else available)
    if not 1 <= count <= available:
        raise ValueError('Requested episode count is outside the committed manifest')
    config = replace(config, data_protocol='base_novel', trials=count, per_class_limit=None)
    result = run(config, storage, scratch, target, episode_manifest=path, prepared=prepared)
    write_json(Path(result['output']) / 'data-audit-reference.json', {'audit_dir': result['data_audit']})
    return {**result, 'report': create_report(Path(result['output']) / 'results.json')}
