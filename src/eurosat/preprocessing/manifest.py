"""Validate the complete split contract, then select fixed episodes for execution."""
from dataclasses import dataclass, replace
import json
from pathlib import Path
import numpy as np

from ..cache import digest_file, key, write_json
from ..config import BASE_CLASSES, CLASSES, NOVEL_CLASSES
from ..evaluation import fit_classifier
from .data import load_source


@dataclass
class ValidatedManifest:
    payload: dict
    sha256: str
    pools: list
    episode_counts: list
    checks: list


@dataclass
class EpisodeData:
    root: Path
    records: list
    split: dict
    dataset_id: str
    trials: list
    episodes: dict
    source: dict
    selection_id: str
    audit_dir: Path


def validate_manifest(path, source):
    """Check every listed file and episode, including those outside a smoke run."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Copy the committed splits_manifest.json to {path}')

    saved = json.loads(path.read_text())
    meta = saved['meta']
    if meta['base_classes'] != BASE_CLASSES or meta['novel_classes'] != NOVEL_CLASSES:
        raise ValueError('Manifest class membership/order differs from this project')
    if meta['shots'] != [1, 5, 10] or meta['n_query'] != 15:
        raise ValueError('Expected 1/5/10-shot episodes and 15 queries per class')

    inventory = {name: {} for name in CLASSES}
    for row in source.records:
        cls, filename = row['path'].split('/', 1)
        inventory[cls][filename] = row['pixels_sha256']

    pools, rows, counts = {}, [], []
    for cls in CLASSES:
        names = ['train', 'val', 'test'] if cls in BASE_CLASSES else ['dev_pool', 'test_pool']
        group = saved['base'] if cls in BASE_CLASSES else saved['novel']
        used = set()

        for name in names:
            filenames = group[name][cls]
            members = set(filenames)
            if len(members) != len(filenames):
                raise ValueError(f'Duplicate filenames in {cls}/{name}')
            if members - inventory[cls].keys():
                raise ValueError(f'Missing or deduplicated files in {cls}/{name}')
            if used & members:
                raise ValueError(f'Image overlap between pools for {cls}')

            used.update(members)
            pools[cls, name] = members
            rows.append({'class': cls, 'role': 'base (reserved)' if cls in BASE_CLASSES else 'novel',
                         'pool': name, 'images': len(members)})

        if used != set(inventory[cls]):
            raise ValueError(f'Manifest does not cover the validated source images for {cls}')

    dev_pixels = {inventory[c][f] for c in NOVEL_CLASSES for f in pools[c, 'dev_pool']}
    test_pixels = {inventory[c][f] for c in NOVEL_CLASSES for f in pools[c, 'test_pool']}
    if dev_pixels & test_pixels:
        raise ValueError('Decoded pixels overlap between novel development and test pools')

    for phase, pool in [('dev', 'dev_pool'), ('test', 'test_pool')]:
        for shots in meta['shots']:
            episodes = saved['episodes'][phase][f'{shots}shot']
            if not episodes:
                raise ValueError(f'Empty {phase}/{shots}-shot episode list')

            for number, episode in enumerate(episodes):
                if set(episode) != set(NOVEL_CLASSES):
                    raise ValueError(f'{phase}/{shots}/{number}: expected four novel classes')
                for cls, parts in episode.items():
                    support, query = parts['support'], parts['query']
                    if (len(support) != shots or len(set(support)) != shots or
                            len(query) != meta['n_query'] or len(set(query)) != meta['n_query'] or
                            set(support) & set(query) or not set(support + query) <= pools[cls, pool]):
                        raise ValueError(f'{phase}/{shots}/{number}/{cls}: invalid support/query assignments')

            counts.append({'phase': phase, 'shots': shots, 'episodes': len(episodes),
                           'support_per_episode': 4 * shots, 'query_per_episode': 4 * meta['n_query']})

    checks = [
        {'check': 'Class order, shot budgets and query count', 'result': 'passed'},
        {'check': 'Full image coverage and unique pool membership', 'result': 'passed'},
        {'check': 'Development/test filename and decoded-pixel isolation', 'result': 'passed'},
        {'check': 'Support/query separation in every committed episode', 'result': 'passed'},
    ]
    return ValidatedManifest(saved, digest_file(path), rows, counts, checks)


def select_episodes(source, validated, config, storage):
    """Resolve committed filenames to stable record indices; never resample or rewrite splits."""
    saved = validated.payload
    if config.seed != saved['meta']['seed']:
        raise ValueError('Use the seed recorded in the committed manifest')

    selected, referenced = {}, set()
    for phase, limit in [('dev', 2 if config.mode == 'smoke' else None), ('test', config.trials)]:
        selected[phase] = {}
        for shots in config.shots:
            episodes = saved['episodes'][phase][f'{shots}shot']
            if limit and not 1 <= limit <= len(episodes):
                raise ValueError('Requested episode count exceeds the committed manifest')
            selected[phase][str(shots)] = episodes[:limit]
            for episode in selected[phase][str(shots)]:
                referenced.update(f'{cls}/{name}' for cls, parts in episode.items()
                                  for role in ['support', 'query'] for name in parts[role])

    available = {row['path']: row for row in source.records}
    records = [available[name] for name in sorted(referenced)]
    lookup = {row['path']: i for i, row in enumerate(records)}

    def indices(episode, role):
        return [lookup[f'{cls}/{name}'] for cls in NOVEL_CLASSES for name in episode[cls][role]]

    episodes = {phase: {k: [{'support': indices(ep, 'support'), 'query': indices(ep, 'query')}
                            for ep in items] for k, items in groups.items()}
                for phase, groups in selected.items()}
    trials = [{str(k): episodes['test'][str(k)][i]['support'] for k in config.shots}
              for i in range(config.trials)]
    split = {phase: sorted({i for items in episodes[phase].values() for ep in items
                            for role in ['support', 'query'] for i in ep[role]}) for phase in ['dev', 'test']}
    provenance = {'sha256': validated.sha256, 'dev_episodes': len(next(iter(episodes['dev'].values()))),
                  'test_episodes': config.trials, 'query_per_class': saved['meta']['n_query'],
                  'selection': 'C chosen on development episodes only; fixed augmentation strengths'}
    dataset_id = key({'source_dataset': source.source_id, 'manifest': validated.sha256, 'records': records})
    selection_id = key({'mode': config.mode, 'seed': config.seed, 'shots': config.shots, 'trials': config.trials})
    audit_dir = Path(storage) / 'cache/data-audits' / validated.sha256 / selection_id
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_json(audit_dir / 'audit.json', {'archive_sha256': source.archive_sha256,
               'manifest_sha256': validated.sha256, 'source_integrity': source.integrity,
               'class_inventory': source.inventory, 'pool_counts': validated.pools,
               'episode_counts': validated.episode_counts, 'checks': validated.checks,
               'selection': provenance, 'selected_unique_images': len(records)})
    write_json(audit_dir / 'selected-records.json', records)
    write_json(audit_dir / 'selected-episodes.json', episodes)

    return EpisodeData(source.root, records, split, dataset_id, trials, episodes,
                       provenance, selection_id, audit_dir)


def prepare_manifest_data(storage, scratch, config, path):
    source = load_source(storage, scratch, config.seed)
    validated = validate_manifest(path, source)
    return select_episodes(source, validated, config, storage)


def select_episode_classifier(original, augmented, aug_map, labels, episodes, shots, policy, config):
    """Choose C on development episodes; ties keep the smaller C."""
    best = None
    for c in sorted(config.classifier_cs):
        accuracies = []
        for episode in episodes['dev'][str(shots)]:
            x, y = episode_support(original, augmented, aug_map, labels, episode['support'], policy, config)
            _, _, accuracy = fit_classifier(x, y, original[episode['query']], labels[episode['query']],
                                             replace(config, classifier_cs=[c]))
            accuracies.append(accuracy)
        candidate = (float(np.mean(accuracies)), c)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best[1], best[0]


def episode_support(original, augmented, aug_map, labels, indices, policy, config):
    x, y = original[indices], labels[indices]
    if policy == 'repeat':
        return np.tile(x, (config.views + 1, 1)), np.tile(y, config.views + 1)
    if augmented is not None:
        additional = [aug_map[i, view] for i in indices for view in range(config.views)]
        return np.concatenate([x, augmented[additional]]), np.concatenate([y, np.repeat(y, config.views)])
    return x, y
