"""The report tables T1-T4 (figure_n_table_sug.md). Each returns (table to read, numbers for CSV)."""
import json

import pandas as pd

from ..config import AUGMENTATION_METHODS, BASELINE_LABELS, MODEL_LABELS, MODEL_PRETRAINING, MODEL_ROLES
from ..models import EMBEDDING_DIMENSIONS, build_backbone, finetuned_files
from .figures import MissingData, baselines_in, conditions_in, gains, models_in, shots_in
from .style import METHOD_LABELS


def _interval(value, low, high, signed=False):
    form = '+.1f' if signed else '.1f'
    value, low, high = (round(v, 1) + 0.0 for v in (value, low, high))  # Avoid "-0.0".
    return format(value, form) if low != low else f'{value:{form}} [{low:{form}}, {high:{form}}]'


def t1_main_results(results):
    """Accuracy (%) per baseline, model and shot without augmentation, with Macro-F1."""
    accuracy, f1 = results.summary(), results.summary('macro_f1')
    accuracy, f1 = accuracy[accuracy.condition == 'none'], f1[f1.condition == 'none']
    numbers = accuracy.merge(f1[['baseline', 'model', 'shots', 'mean']].rename(columns={'mean': 'macro_f1'}),
                             on=['baseline', 'model', 'shots']).drop(columns='condition')
    numbers = numbers.rename(columns={'mean': 'accuracy'})
    rows = []
    for baseline in baselines_in(results):
        for model in models_in(results):
            row = {'Baseline': BASELINE_LABELS[baseline], 'Model': MODEL_LABELS[model]}
            for shot in shots_in(results):
                match = numbers[(numbers.baseline == baseline) & (numbers.model == model) & (numbers.shots == shot)]
                if len(match):
                    r = match.iloc[0]
                    row[f'{shot}-shot accuracy (%)'] = _interval(r.accuracy, r.ci_low, r.ci_high)
                    row[f'{shot}-shot Macro-F1 (%)'] = f'{r.macro_f1:.1f}'
            rows.append(row)
    return pd.DataFrame(rows).set_index(['Baseline', 'Model']), numbers


def t2_models(results, weights_dir):
    """Model size, compute cost (embeddings, classifiers, GPU memory) and the fine-tuning outcome.

    Costs come from the pretrained backbone (or another baseline if it was not computed in this run).
    """
    rows, numbers = [], []
    for model in models_in(results):
        costs = [c for c in results.meta.get('costs', []) if c['model'] == model]
        timed = next((c for c in costs if c['baseline'] == 'pretrained'), costs[0] if costs else {})
        parameters = next((c['parameters'] for c in costs if c.get('parameters')), None)
        if parameters is None:
            parameters = sum(p.numel() for p in build_backbone(model, 'untrained').parameters())
        speed = timed['computed_images'] / timed['computed_seconds'] if timed.get('computed_seconds') else None
        memory = timed.get('embedding_memory', {}).get('cuda_peak_allocated_bytes')
        record_path = finetuned_files(model, weights_dir)[1]
        record = json.loads(record_path.read_text()) if record_path.is_file() else {}
        number = {'model': model, 'parameters_millions': parameters / 1e6, 'embedding_dim': EMBEDDING_DIMENSIONS[model],
                  'images_per_second': speed, 'classifier_fits': timed.get('fits'),
                  'classifier_fit_seconds': timed.get('fit_seconds'),
                  'peak_gpu_memory_gb': memory / 1e9 if memory else None,
                  'finetune_learning_rate': record.get('chosen_learning_rate'),
                  'finetune_val_accuracy': record.get('val_accuracy'), 'finetune_test_accuracy': record.get('test_accuracy'),
                  'finetune_minutes': sum(run['minutes'] for run in record.get('runs', [])) or None}
        numbers.append(number)
        rows.append({'Model': MODEL_LABELS[model], 'Type': MODEL_ROLES[model], 'Pretraining': MODEL_PRETRAINING[model],
                     'Params (M)': f'{number["parameters_millions"]:.1f}', 'Embedding': number['embedding_dim'],
                     'Images / s': f'{speed:,.0f}' if speed else '—',
                     'Classifier fitting (s)': (f'{number["classifier_fit_seconds"]:,.0f}'
                                                if number['classifier_fits'] else '—'),
                     'Peak GPU memory (GB)': f'{number["peak_gpu_memory_gb"]:.1f}' if memory else '—',
                     'Fine-tuning learning rate': f'{record["chosen_learning_rate"]:g}' if record else '—',
                     'Val accuracy, 6 classes (%)': f'{record["val_accuracy"] * 100:.1f}' if record else '—',
                     'Test accuracy, 6 classes (%)': f'{record["test_accuracy"] * 100:.1f}' if record else '—',
                     'Fine-tuning (min)': f'{number["finetune_minutes"]:.0f}' if number['finetune_minutes'] else '—'})
    return pd.DataFrame(rows).set_index('Model'), pd.DataFrame(numbers)


def t3_preferred_augmentation(results):
    """Per baseline, model and shot: the single method chosen on development episodes, checked on test."""
    methods = [m for m in AUGMENTATION_METHODS if m in conditions_in(results)]
    if not methods:
        raise MissingData('needs track 3')
    dev = results.groups.set_index(['baseline', 'model', 'condition', 'shots'])['dev_accuracy']
    test = gains(results, methods).set_index(['condition', 'baseline', 'model', 'shots'])
    rows, numbers = [], []
    for baseline in baselines_in(results):
        for model in models_in(results):
            for shot in shots_in(results):
                if (baseline, model, 'none', shot) not in dev.index:
                    continue
                reference = dev[(baseline, model, 'none', shot)]
                scores = {m: dev[(baseline, model, m, shot)] for m in methods if (baseline, model, m, shot) in dev.index}
                best = max(scores, key=lambda m: (scores[m], -methods.index(m)))
                chosen = best if scores[best] > reference else 'none'
                number = {'baseline': baseline, 'model': model, 'shots': shot, 'method': chosen,
                          'dev_gain': (scores[best] - reference) * 100 if chosen != 'none' else None}
                row = {'Baseline': BASELINE_LABELS[baseline], 'Model': MODEL_LABELS[model], 'Shots': shot,
                       'Dev-selected method': METHOD_LABELS.get(chosen, 'none (no method beat no augmentation)')}
                if chosen == 'none':
                    row.update({'Dev gain (pp)': '—', 'Test gain (pp) [95% CI]': '—', 'Episodes with no gain': '—'})
                else:
                    r = test.loc[(chosen, baseline, model, shot)]
                    number.update(test_gain=r.difference, ci_low=r.ci_low, ci_high=r.ci_high, share_not_better=r.share_not_better)
                    row.update({'Dev gain (pp)': f'{number["dev_gain"]:+.1f}',
                                'Test gain (pp) [95% CI]': _interval(r.difference, r.ci_low, r.ci_high, signed=True),
                                'Episodes with no gain': f'{r.share_not_better * 100:.0f}%'})
                rows.append(row)
                numbers.append(number)
    return pd.DataFrame(rows).set_index(['Baseline', 'Model', 'Shots']), pd.DataFrame(numbers)


def t4_labels_used(results, weights_dir):
    """Labeled images by role. The shot budget fits each classifier; development labels only choose settings."""
    usage = results.meta.get('labels')
    if not usage:
        raise MissingData('needs a results file that records label use (rerun Section 5)')
    rows, numbers = [], []
    for k in [*map(str, shots_in(results)), 'all']:
        dev, test = usage['dev'][k], usage['test'][k]
        rows.append({'Labels': f'{k}-shot' if k != 'all' else 'All shots',
                     'Train (per classifier)': (f'{test["support_per_episode"]} support images'
                                                if k != 'all' else '—'),
                     'Choose settings': f'{dev["distinct_images"]:,} development images, {dev["episodes"]} episodes',
                     'Score only': f'{test["distinct_query"]:,} test images, {test["episodes"]} episodes'})
        numbers += [{'shots': k, 'phase': phase, **usage[phase][k]} for phase in ['dev', 'test']]
    record = next((json.loads(path.read_text()) for model in models_in(results)
                   if 'finetuned' in baselines_in(results)
                   and (path := finetuned_files(model, weights_dir)[1]).is_file()), None)
    if record:
        images = record['images']
        rows.append({'Labels': 'Fine-tuning (6 other classes)', 'Train (per classifier)': f'{images["train"]:,} train images',
                     'Choose settings': f'{images["val"]:,} val images (learning rate, epoch)',
                     'Score only': f'{images["test"]:,} test images (a check)'})
        numbers.append({'shots': 'fine-tuning', 'phase': 'base classes', **images})
    return pd.DataFrame(rows).set_index('Labels'), pd.DataFrame(numbers)
