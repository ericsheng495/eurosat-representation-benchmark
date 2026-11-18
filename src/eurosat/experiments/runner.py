"""Run the benchmark: every baseline x model x condition x shot on the same fixed episodes.

For each backbone (baseline x model):
  1. skip it if every result group is already saved (the model is not even loaded)
  2. compute or load embeddings: originals for all selected images, augmented views for support images only
  3. for each condition and shot: choose C on development episodes, score the test episodes, save one file
  4. save its compute cost to <run>/costs/, so a resumed or repeated run keeps it
At the end, results.json lists every group with its key numbers, the costs and how many labels each role used;
per-episode confusion matrices stay in the group files.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import time

import numpy as np
import torch
from joblib import Parallel
from joblib.externals.loky import get_reusable_executor

from ..cache import key, prepare_storage, write_json, write_npz
from ..config import CLASSES, config_dict
from ..models import backbone_identity, load_backbone
from ..preprocessing.manifest import selection_id
from ..runtime import check_device, environment, memory_usage
from ..training import finetuned_weights_dir
from .classifier import evaluate_condition
from .conditions import build_conditions, needed_views
from .embeddings import PREPROCESS, EmbeddingStore


def support_images(episodes, shots):
    """Support images of the development and test episodes at these shot budgets (augmentation is support-only)."""
    return sorted({i for phase in ['dev', 'test'] for k in shots for ep in episodes[phase][str(k)] for i in ep['support']})


def label_usage(episodes, shots):
    """Labeled images per role and shot. Development labels choose settings; test supports are the shot budget."""
    usage = {}
    for phase in ['dev', 'test']:
        usage[phase] = {}
        for k in [*map(str, shots), 'all']:
            chosen = [ep for s in shots for ep in episodes[phase][str(s)]] if k == 'all' else episodes[phase][k]
            support = {i for ep in chosen for i in ep['support']}
            query = {i for ep in chosen for i in ep['query']}
            usage[phase][k] = {'episodes': len(chosen), 'distinct_support': len(support),
                               'distinct_query': len(query), 'distinct_images': len(support | query)}
            if k != 'all':
                usage[phase][k].update(support_per_episode=len(chosen[0]['support']),
                                       query_per_episode=len(chosen[0]['query']))
    return usage


def run(config, storage, scratch, manifest_path, prepared, target='local', device=None):
    """Run or resume. Returns the output folder and the path of results.json."""
    config.validate()
    storage = prepare_storage(storage)
    if prepared.selection_id != selection_id(config):
        raise ValueError('The checked episodes belong to different settings: rerun the data check (Section 3)')
    device = device or check_device(target)
    plan = build_conditions(config.tracks, config.view_sweep, config.shots)
    views_needed = needed_views(plan)
    weights_dir = finetuned_weights_dir(storage, smoke=config.mode == 'smoke')
    records, episodes = prepared.records, prepared.episodes
    labels = np.array([row['label'] for row in records])
    label_order = sorted(set(labels.tolist()))
    total = sum(len(shots) for _, shots in plan)

    pairs = [(baseline, model) for baseline in config.baselines for model in config.models]
    backbones = [backbone_identity(model, baseline, weights_dir) for baseline, model in pairs]
    run_id = key({'config': config_dict(config), 'dataset': prepared.dataset_id, 'selection': prepared.selection_id,
                  'backbones': backbones})
    output = storage / 'outputs' / run_id
    (output / 'costs').mkdir(parents=True, exist_ok=True)
    print(f'{len(config.baselines)} baselines x {len(config.models)} models x {total} condition-shot groups', flush=True)

    groups = []
    # Episodes are small, so arrays are passed directly (max_nbytes=None) instead of through temporary files.
    with Parallel(n_jobs=config.jobs, max_nbytes=None) as parallel:
        for (baseline, model), backbone in zip(pairs, backbones):
            todo = []
            for condition, shots in plan:
                params = {m: config.augmentation[m] for m in condition.methods}
                for shot in shots:
                    identity = {'backbone': backbone, 'dataset': prepared.dataset_id, 'selection': prepared.selection_id,
                                'condition': asdict(condition), 'shot': shot, 'augmentation': params,
                                'cs': sorted(config.classifier_cs), 'seed': config.seed, 'preprocess': PREPROCESS}
                    path = storage / 'cache/results' / f'{key(identity)}.npz'
                    groups.append({'baseline': baseline, 'model': model, 'condition': condition.name,
                                   'track': condition.track, 'control': condition.control, 'rows': condition.rows,
                                   'shots': shot, 'file': path.relative_to(storage).as_posix()})
                    if not path.is_file():
                        todo.append((condition, shot, path))
            print(f'\n{baseline} · {model}: {total - len(todo)} of {total} groups saved earlier, '
                  f'{len(todo)} to compute', flush=True)
            if not todo:
                continue

            started = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            store = EmbeddingStore(storage, backbone, model,
                                   lambda m=model, b=baseline: load_backbone(m, device, b, weights_dir),
                                   prepared, config, device)
            # Originals of every selected image, plus each needed augmented view of the support images.
            items = sorted({item for condition, _, _ in todo for item in condition.views})
            supports = [support_images(episodes, views_needed[item]) for item in items]
            arrays = store.get_many([(np.arange(len(records)), 'none', 0)] +
                                    [(images, *item) for item, images in zip(items, supports)])
            original = arrays[0]
            views = {item: (array, {image: row for row, image in enumerate(images)})
                     for item, images, array in zip(items, supports, arrays[1:])}
            parameters = sum(p.numel() for p in store.model.parameters()) if store.model is not None else None
            store.release()
            cost = {'baseline': baseline, 'model': model, 'parameters': parameters, **store.stats,
                    'embedding_memory': memory_usage(device), 'fit_seconds': 0.0, 'fits': 0}
            print(f'  embeddings: {len(arrays)} sets ({store.stats["cached_sets"]} reused), '
                  f'{store.stats["computed_images"]:,} images computed in {time.perf_counter() - started:.0f}s', flush=True)

            for condition, shot, path in todo:
                result = evaluate_condition(parallel, original, views, labels, label_order, episodes,
                                            condition, shot, config.classifier_cs, config.seed)
                write_npz(path, **result)
                cost['fit_seconds'] += result['fit_seconds']
                cost['fits'] += result['fits']
            print(f'  classifiers: {len(todo)} groups, {cost["fits"]:,} fits in {cost["fit_seconds"]:.0f}s', flush=True)
            cost['total_seconds'] = time.perf_counter() - started
            write_json(output / 'costs' / f'{baseline}_{model}.json', cost)
            del original, views
    get_reusable_executor().shutdown(wait=True)  # Stop the parallel workers so no process outlives the run.

    for entry in groups:  # Key numbers for every group, including those saved by earlier runs.
        with np.load(storage / entry['file']) as saved:
            confusion = saved['confusion']
            entry.update({'c': float(saved['c']), 'c_at_edge': bool(saved['c_at_edge']),
                          'dev_accuracy': float(saved['dev_accuracy_by_c'].max()),
                          'test_accuracy': float(np.trace(confusion, axis1=1, axis2=2).sum() / confusion.sum()),
                          'episodes': int(confusion.shape[0]), 'nonconverged_fits': int(saved['nonconverged_fits'])})

    cost_files = [output / 'costs' / f'{baseline}_{model}.json' for baseline, model in pairs]
    results = {'config': config_dict(config), 'run_id': run_id, 'dataset_id': prepared.dataset_id,
               'selection_id': prepared.selection_id, 'episode_source': prepared.source,
               'classes': [CLASSES[label] for label in label_order], 'backbones': backbones,
               'groups': groups, 'labels': label_usage(episodes, config.shots),
               'costs': [json.loads(path.read_text()) for path in cost_files if path.is_file()],
               'environment': environment(), 'device': str(device),
               'hardware': torch.cuda.get_device_name(0) if device.type == 'cuda' else str(device),
               'finished_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    write_json(output / 'results.json', results)
    print(f'\nFinished. Results: {output / "results.json"}', flush=True)
    return {'output': str(output), 'results_path': str(output / 'results.json')}
