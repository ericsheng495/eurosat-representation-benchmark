"""Proposal-aligned Part D: development selection, main grid and view-budget study."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np

from .cache import key, prepare_storage, write_json
from .config import Config, MODEL_IDS, POLICIES
from .data import make_trials, prepare_data
from .evaluation import fit_classifier
from .features import extract
from .models import load_backbone
from .pipeline import run
from .reporting import compare_budgets, create_report, merge_results
from .runtime import check_device, environment


def part_d_config(mode='full', include_cutout=False):
    """The primary DINOv2 grid; smoke preserves every condition on a small subset."""
    if mode not in {'full', 'smoke'}:
        raise ValueError('mode must be full or smoke')
    config = Config.full() if mode == 'full' else Config(trials=2, views=4, policies=POLICIES.copy())
    config.models = ['dinov2_vits14']
    if include_cutout:
        config.policies.append('cutout')
    config.validate()
    return config


def tune_strengths(config, storage, scratch, target='colab_gpu', candidates=(.5, 1., 1.5)):
    """Choose each policy's strength at 1-shot using development labels only."""
    config.validate()
    if len(config.models) != 1:
        raise ValueError('Tune with one designated backbone, then share the locked strengths')
    if not candidates or any(not 0 < strength <= 2 for strength in candidates):
        raise ValueError('Supply strength candidates in (0, 2]')
    storage = prepare_storage(storage)
    root, records, split, dataset_id = prepare_data(storage, scratch, config)
    tuning = replace(config, shots=[1], policy_strengths={})
    trials = make_trials(records, split, tuning)
    manifest = {'purpose': 'development-only-strength-selection-v1', 'dataset': dataset_id,
                'config': asdict(tuning), 'trials': trials, 'candidates': list(candidates),
                'checkpoint': MODEL_IDS[config.models[0]], 'versions': environment()}
    directory = storage / 'cache/development' / key(manifest)
    directory.mkdir(parents=True, exist_ok=True)
    completed = directory / 'selection.json'
    if completed.exists():
        print('Development strength selection reused:', completed, flush=True)
        return json.loads(completed.read_text())
    write_json(directory / 'manifest.json', manifest)
    device = check_device(target)
    model = None
    original = None
    labels = np.array([r['label'] for r in records])
    support_union = sorted({i for trial in trials for i in trial['1']})
    selected, scores = {}, []
    for policy in config.policies:
        if policy in {'none', 'repeat'}:
            continue
        strengths = [1.] if policy == 'flip' else sorted(set(candidates))
        options = []
        for strength in strengths:
            accuracies = []
            augmented, lookup = None, {}
            for number, trial in enumerate(trials):
                path = directory / (key([policy, strength, number]) + '.json')
                if path.exists():
                    result = json.loads(path.read_text())
                else:
                    if model is None:
                        model = load_backbone(config.models[0], device)
                        # Frozen query embeddings may be cached here, but query labels
                        # and predictions never enter development selection.
                        original, _ = extract(storage, root, records, dataset_id, config.models[0], model,
                                              device, [(i, 0) for i in range(len(records))], 'none', tuning)
                    candidate = replace(tuning, policy_strengths={policy: strength})
                    if augmented is None:
                        items = [(i, view) for i in support_union for view in range(config.views)]
                        augmented, _ = extract(storage, root, records, dataset_id, config.models[0], model,
                                               device, items, policy, candidate)
                        lookup = {item: j for j, item in enumerate(items)}
                    indices = trial['1']
                    additional = [lookup[i, view] for i in indices for view in range(config.views)]
                    x = np.concatenate([original[indices], augmented[additional]])
                    y = np.concatenate([labels[indices], np.repeat(labels[indices], config.views)])
                    _, c, accuracy = fit_classifier(x, y, original[split['dev']], labels[split['dev']], candidate)
                    result = {'policy': policy, 'strength': strength, 'trial': number,
                              'C': c, 'dev_accuracy': accuracy}
                    write_json(path, result)
                accuracies.append(result['dev_accuracy'])
                scores.append(result)
            options.append({'strength': strength, 'dev_accuracy': float(np.mean(accuracies))})
        # Prefer the smaller strength on a development tie; query scores are absent.
        selected[policy] = sorted(options, key=lambda row: (-row['dev_accuracy'], row['strength']))[0]['strength']
        print(f'Development selection: {policy} strength={selected[policy]}', flush=True)
    del model
    if device.type == 'mps':
        import torch
        torch.mps.empty_cache()
    result = {'manifest': manifest, 'selected_strengths': selected, 'scores': scores,
              'development_labels': len(split['dev']), 'query_labels_used': 0,
              'selection_shots': 1, 'output': str(directory)}
    write_json(completed, result)
    return result


def select_nine_view_methods(primary, count=2):
    """Prespecify the follow-up methods from primary development performance."""
    if count < 1:
        raise ValueError('Select at least one nine-view method')
    candidates = [row for row in primary['summary'] if row['shots'] == 1
                  and row['policy'] not in {'none', 'repeat'}]
    if not candidates:
        raise ValueError('Primary grid must contain 1-shot augmentation conditions')
    return [row['policy'] for row in sorted(candidates,
            key=lambda row: (-row['dev_accuracy'], row['policy']))[:count]]


def run_part_d(storage, scratch, target='colab_gpu', mode='full', *,
               select_strengths=True, include_cutout=False, selected_methods=2, config=None, include_resnet=True):
    """Run and resume all required DINOv2 comparisons without touching GitHub."""
    if selected_methods < 1:
        raise ValueError('Select at least one nine-view method')
    config = config or part_d_config(mode, include_cutout)
    config.validate()
    if config.models != ['dinov2_vits14'] or config.views != 4 or set(config.shots) != {1, 5, 10} or not set(POLICIES) <= set(config.policies):
        raise ValueError('Part D primary grid needs DINOv2, all ten conditions, 1/5/10 shots and four views')
    mode = config.mode
    storage = prepare_storage(storage)
    _, _, _, dataset_id = prepare_data(storage, scratch, config)
    suite_id = key({'suite': 'part-d-v1', 'dataset': dataset_id, 'config': asdict(config),
                    'include_resnet_baseline': include_resnet, 'strength_selection': select_strengths, 'strength_candidates': [.5, 1., 1.5],
                    'selected_methods': selected_methods,
                    'versions': environment()})
    directory = storage / 'outputs/part-d' / suite_id
    directory.mkdir(parents=True, exist_ok=True)
    plan = {'mode': mode, 'primary': asdict(config), 'include_resnet_baseline': include_resnet, 'strength_selection': select_strengths,
            'strength_candidates': [.5, 1., 1.5], 'strength_selection_shots': 1,
            'nine_view_method_selection': '1-shot primary development accuracy; name tie-break',
            'selected_methods': selected_methods, 'dataset': dataset_id}
    write_json(directory / 'plan.json', plan)
    started = time.perf_counter()
    if select_strengths:
        tuning = tune_strengths(config, storage, scratch, target)
        config = replace(config, policy_strengths=tuning['selected_strengths'])
        write_json(directory / 'strength-selection.json', tuning)
    primary = run(config, storage, scratch, target)
    primary_file = Path(primary['output']) / 'results.json'
    primary_report = create_report(primary_file)
    resnet, combined, comparison_report = None, None, None
    if include_resnet:
        # Identical strengths, original supports, splits and classifier rules.
        resnet = run(replace(config, models=['resnet18']), storage, scratch, target)
        combined = merge_results([primary_file, Path(resnet['output']) / 'results.json'], directory / 'model-baseline')
        comparison_report = create_report(Path(combined['output']) / 'results.json')
    methods = select_nine_view_methods(primary, selected_methods)
    # This artifact is written before evaluating any nine-view query predictions.
    write_json(directory / 'nine-view-selection.json', {'policies': methods, 'selected_using': 'development accuracy'})
    followup = replace(config, shots=[1], policies=['none', 'repeat', *methods], views=9)
    nine = run(followup, storage, scratch, target)
    nine_file = Path(nine['output']) / 'results.json'
    nine_report = create_report(nine_file)
    comparisons = compare_budgets(primary_file, nine_file, directory / 'comparisons')
    result = {'mode': mode, 'output': str(directory), 'primary': primary, 'nine_view': nine,
              'primary_report': primary_report, 'nine_view_report': nine_report,
              'resnet_baseline': resnet, 'combined': combined, 'comparison_report': comparison_report,
              'comparisons': comparisons, 'selected_methods': methods,
              'seconds_this_execution': time.perf_counter() - started}
    write_json(directory / 'suite.json', result)
    print('Part D suite complete:', directory, flush=True)
    return result
