"""Experiment orchestration; implementation lives in focused modules."""
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import io
import json
import time
import numpy as np
import torch
from .cache import atomic, key, prepare_storage, write_json, write_npz
from .config import Config, CLASSES, MODEL_IDS
from .data import prepare_data, make_trials
from .evaluation import complete_metrics, fit_classifier, score, summarize
from .features import extract
from .models import load_backbone
from .runtime import check_device, environment


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
    execution_path = directory / f'execution-{time.time_ns()}.json'
    execution = {'utc': datetime.now(timezone.utc).isoformat(), 'device': str(device),
                 'hardware': hardware, 'target': target, 'versions': environment(), 'stages': []}
    write_json(execution_path, execution)
    labels_all = np.array([r['label'] for r in records])
    all_items = [(i, 0) for i in range(len(records))]
    support_union = sorted({i for trial in trials for indices in trial.values() for i in indices})
    def load_result(path):
        row = json.loads(path.read_text())
        previous = dict(row['test'])
        complete_metrics(row)
        # Enrich an existing completed trial while retaining its classifier and features.
        if row['test'] != previous:
            write_json(path, row)
        return row

    rows = []
    for model_name in config.models:
        # A fully completed model is skipped without loading its weights or features.
        result_paths = [directory / (key([model_name, p, trial, k]) + '.json')
                        for p in config.policies for trial in range(config.trials) for k in config.shots]
        if all(p.exists() for p in result_paths):
            rows.extend(load_result(p) for p in result_paths)
            print('Completed results reused:', model_name, flush=True)
            continue
        started = time.perf_counter()
        model = load_backbone(model_name, device)
        model_load_seconds = time.perf_counter() - started
        stage = {'stage': 'original_features', 'model_load_seconds': model_load_seconds}
        original, _ = extract(storage, root, records, dataset_id, model_name, model,
                               device, all_items, 'none', config, stats=stage)
        execution['stages'].append(stage)
        write_json(execution_path, execution)
        feature_seconds = time.perf_counter() - started
        for policy in config.policies:
            paths = [directory / (key([model_name, policy, trial, k]) + '.json')
                     for trial in range(config.trials) for k in config.shots]
            if all(path.exists() for path in paths):
                rows.extend(load_result(path) for path in paths)
                continue
            augmented = None
            aug_map = {}
            if policy not in {'none', 'repeat'}:
                items = [(i, v) for i in support_union for v in range(config.views)]
                stage = {'stage': 'augmented_features'}
                augmented, _ = extract(storage, root, records, dataset_id, model_name, model,
                                       device, items, policy, config, stats=stage)
                execution['stages'].append(stage)
                write_json(execution_path, execution)
                aug_map = {item: j for j, item in enumerate(items)}
            for trial_number, trial in enumerate(trials):
                for k in config.shots:
                    result_path = directory / (key([model_name, policy, trial_number, k]) + '.json')
                    if result_path.exists():
                        rows.append(load_result(result_path))
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
                              'parameter_count': sum(p.numel() for p in model.parameters()),
                              'model_tensor_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
                              'computed_on': {'device': str(device), 'hardware': hardware, 'execution': execution_path.name}}
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
    write_json(output / 'results.json', {'manifest': manifest, 'trials': rows, 'summary': summary,
                                        'executions': [json.loads(p.read_text()) for p in sorted(directory.glob('execution-*.json'))]})
    columns = list(summary[0])
    stream = io.StringIO()
    writer = csv.DictWriter(stream, lineterminator='\n', fieldnames=columns)
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
