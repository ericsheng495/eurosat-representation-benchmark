"""Combine compatible teammate exports without mixing experimental protocols."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path

from .cache import atomic, digest_file, key, write_json
from .config import CLASSES, MODEL_IDS
from .evaluation import summarize
from .exploration import STYLE, save_figure


def comparison_identity(manifest):
    comparable = deepcopy(manifest)
    comparable['config'].pop('models')
    comparable.pop('checkpoints')
    return key(comparable)


def merge_results(inputs, output):
    """Merge completed model assignments with identical data, trials and settings."""
    if not inputs:
        raise ValueError('Supply at least one results.json export')
    reports = [json.loads(Path(path).read_text()) for path in inputs]
    identity = comparison_identity(reports[0]['manifest'])
    rows, seen, checkpoints = [], set(), {}
    for report in reports:
        manifest = report['manifest']
        if comparison_identity(manifest) != identity:
            raise ValueError('Cannot combine different datasets, trials, software versions or non-model settings')
        config = manifest['config']
        if set(manifest['checkpoints']) != set(config['models']):
            raise ValueError('Checkpoint identities must cover exactly the exported models')
        expected = {(model, policy, shot, trial) for model in config['models']
                    for policy in config['policies'] for shot in config['shots']
                    for trial in range(config['trials'])}
        received = [(r['model'], r['policy'], r['shots'], r['trial']) for r in report['trials']]
        if len(received) != len(expected) or set(received) != expected:
            raise ValueError('A teammate export contains incomplete or duplicated trials')
        if seen & set(received):
            raise ValueError('The same model/policy/shot/trial appears in multiple exports')
        for model, checkpoint in manifest['checkpoints'].items():
            if MODEL_IDS.get(model) != checkpoint:
                raise ValueError('Unrecognized checkpoint identity for ' + model)
        checkpoints.update(manifest['checkpoints'])
        seen.update(received)
        rows.extend(report['trials'])
    manifest = deepcopy(reports[0]['manifest'])
    manifest['config']['models'] = [m for m in MODEL_IDS if m in checkpoints]
    manifest['checkpoints'] = checkpoints
    summary = summarize(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'results.json', {'manifest': manifest, 'trials': rows, 'summary': summary,
                                            'executions': [entry for report in reports for entry in report.get('executions', [])]})
    stream = io.StringIO()
    writer = csv.DictWriter(stream, lineterminator='\n', fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
    atomic(output / 'summary.csv', lambda p: p.write_text(stream.getvalue()))
    print(f'Merged {len(rows)} trials across {len(checkpoints)} models into {output}', flush=True)
    return {'output': str(output), 'summary': summary, 'models': manifest['config']['models']}


def create_report(results, output=None):
    """Render saved trials without model inference or selection on test scores."""
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    source = Path(results)
    report = json.loads(source.read_text())
    manifest, rows = report['manifest'], report['trials']
    config = manifest['config']
    if not rows:
        raise ValueError('Report needs completed trials')
    expected = {(m, p, k, t) for m in config['models'] for p in config['policies']
                for k in config['shots'] for t in range(config['trials'])}
    actual = [(r['model'], r['policy'], r['shots'], r['trial']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Report needs complete, unique paired trials')
    if set(config['models']) - MODEL_IDS.keys() or manifest['checkpoints'] != {m: MODEL_IDS[m] for m in config['models']}:
        raise ValueError('Report checkpoint identities do not match the supported backbones')
    output = Path(output) if output else source.parent / 'report'
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{
        'model': r['model'], 'policy': r['policy'], 'shots': r['shots'], 'trial': r['trial'],
        'accuracy': r['test']['accuracy'], 'macro_f1': r['test']['macro_f1'],
        'dev_accuracy': r['dev_accuracy'], 'fit_seconds': r['classifier_fit_seconds'],
        'extraction_seconds': r['original_feature_seconds'], 'parameters': r['parameter_count']}
        for r in rows])
    baseline = frame[frame.policy == 'none'][['model', 'shots', 'trial', 'accuracy']]
    paired = frame.merge(baseline, on=['model', 'shots', 'trial'], suffixes=('', '_baseline'),
                         validate='many_to_one')
    paired['gain_pp'] = (paired.accuracy - paired.accuracy_baseline) * 100
    summary = summarize(rows)
    selected = [sorted([r for r in summary if r['model'] == m and r['shots'] == k],
                       key=lambda r: (-r['dev_accuracy'], r['policy']))[0]
                for m in config['models'] for k in sorted(config['shots'])]
    write_json(output / 'development-selected-policies.json', selected)
    write_json(output / 'provenance.json', {
        'manifest': manifest, 'source_results_sha256': digest_file(source),
        'executions': report.get('executions', []),
        'timing_note': 'Extraction stage values include cold loading or cache reuse; do not sum repeated trial values. '
                       'Execution sessions may reuse previous work and do not identify every trial computation device.'})
    for name, table in [('summary', pd.DataFrame(summary)), ('paired-trials', paired),
                        ('timings', frame.groupby('model', sort=False).agg(
                            classifier_fit_seconds_mean=('fit_seconds', 'mean'),
                            extraction_stage_seconds_min=('extraction_seconds', 'min'),
                            extraction_stage_seconds_max=('extraction_seconds', 'max'),
                            parameters=('parameters', 'first')).reset_index())]:
        atomic(output / f'{name}.csv', lambda path, table=table: table.to_csv(path, index=False))

    smoke = config['mode'] == 'smoke'
    label = 'SMOKE CHECK — execution validation' if smoke else 'Frozen representations'
    names = ['model_by_shot', 'augmentation_gains', 'class_recall', 'confusion_matrix']
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 3.6), layout='constrained')
        for model in config['models']:
            group = frame[(frame.model == model) & (frame.policy == 'none')]
            stats = group.groupby('shots').accuracy.agg(['mean', lambda s: np.std(s, ddof=0)])
            ax.errorbar(stats.index, stats['mean'] * 100, yerr=stats.iloc[:, 1] * 100,
                        marker='o', capsize=3, label=model)
        ax.set(xlabel='Original labeled images per class', ylabel='Held-out accuracy (%)',
               ylim=(0, 100), xticks=sorted(config['shots']), title=label + '\nUnaugmented baseline; mean ± trial SD')
        ax.legend(frameon=False)
        save_figure(fig, output, names[0])

        fig, axes = plt.subplots(len(config['models']), len(config['shots']),
                                 figsize=(4 * len(config['shots']), 3.4 * len(config['models'])),
                                 squeeze=False, layout='constrained')
        for row_index, model in enumerate(config['models']):
            for col_index, shots in enumerate(sorted(config['shots'])):
                ax = axes[row_index, col_index]
                group = paired[(paired.model == model) & (paired.shots == shots) & (paired.policy != 'none')]
                stats = group.groupby('policy').gain_pp.agg(['mean', lambda s: np.std(s, ddof=0)])
                order = [p for p in config['policies'] if p != 'none']
                stats = stats.reindex(order)
                ax.barh(order, stats['mean'], xerr=stats.iloc[:, 1], color='#376D95', capsize=2)
                ax.axvline(0, color='#687682', linewidth=.8)
                ax.set(title=f'{model} · {shots}-shot', xlabel='Paired accuracy gain (percentage points)')
                ax.invert_yaxis()
        fig.suptitle(label + '\nAll prespecified conditions; mean ± paired-trial SD')
        save_figure(fig, output, names[1])

        highest = max(config['shots'])
        fig, ax = plt.subplots(figsize=(8, 4.2), layout='constrained')
        width = .8 / len(config['models'])
        positions = np.arange(len(CLASSES))
        for index, model in enumerate(config['models']):
            values = np.array([r['test']['class_recall'] for r in rows
                               if r['model'] == model and r['shots'] == highest and r['policy'] == 'none'])
            ax.bar(positions + (index - (len(config['models']) - 1) / 2) * width,
                   values.mean(0) * 100, yerr=values.std(0) * 100, width=width, label=model, capsize=2)
        ax.set(xticks=positions, xticklabels=CLASSES, ylabel='Held-out recall (%)', ylim=(0, 100),
               title=f'{label}\n{highest}-shot unaugmented baseline; mean ± trial SD')
        ax.tick_params(axis='x', labelrotation=45)
        for tick in ax.get_xticklabels():
            tick.set_ha('right')
        ax.legend(frameon=False)
        save_figure(fig, output, names[2])

        fig, axes = plt.subplots(1, len(config['models']), figsize=(6 * len(config['models']), 5),
                                 squeeze=False, layout='constrained')
        for ax, model in zip(axes.flat, config['models']):
            counts = np.sum([r['test']['confusion_matrix'] for r in rows
                             if r['model'] == model and r['shots'] == highest and r['policy'] == 'none'], axis=0)
            normalized = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
            sns.heatmap(normalized, vmin=0, vmax=1, cmap='Blues', ax=ax,
                        xticklabels=CLASSES, yticklabels=CLASSES, cbar=True)
            ax.set(title=f'{model} · {highest}-shot baseline', xlabel='Predicted class', ylabel='True class')
            ax.tick_params(axis='x', labelrotation=90)
        fig.suptitle(label + '\nRow-normalized counts summed across repeated query evaluations')
        save_figure(fig, output, names[3])
    write_json(output / 'report.json', {'mode': config['mode'], 'trials': len(rows),
               'models': config['models'], 'figures': names, 'variability': 'Population standard deviation across paired trials; not a confidence interval'})
    print('Result figures and tables:', output, flush=True)
    return {'output': str(output), 'figures': {name: str(output / f'{name}.png') for name in names},
            'selected': selected}
