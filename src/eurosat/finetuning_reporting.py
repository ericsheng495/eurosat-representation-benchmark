"""Tables and figures for the optional DINOv2 adaptation study."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CLASSES
from .exploration import STYLE, save_figure


def finetuning_contrasts(rows):
    """Paired effects: adaptation, augmentation and additional real labels."""
    indexed = {(r['strategy'], r.get('augmentation', 'single'), r['shots'], r['trial']): r for r in rows}
    contrasts = []
    def compare(kind, candidate, baseline):
        if candidate not in indexed or baseline not in indexed:
            return
        a, b = indexed[candidate], indexed[baseline]
        contrasts.append({'comparison': kind, 'strategy': candidate[0], 'augmentation': candidate[1],
                          'shots': candidate[2], 'trial': candidate[3],
                          'accuracy_gain_pp': 100 * (a['test']['accuracy'] - b['test']['accuracy']),
                          'macro_f1_gain_pp': 100 * (a['test']['macro_f1'] - b['test']['macro_f1'])})
    for strategy, policy, shots, trial in sorted(indexed):
        if strategy == 'last_block':
            compare('adaptation over frozen head', (strategy, policy, shots, trial), ('frozen_head', policy, shots, trial))
        if policy == 'flip':
            for reference in ['none', 'repeat']:
                compare('flip over ' + reference, (strategy, policy, shots, trial), (strategy, reference, shots, trial))
            if shots == 1 and indexed[strategy, policy, shots, trial].get('added_views') == 4:
                compare('1 original + 4 flips over 5 originals', (strategy, policy, shots, trial), (strategy, 'none', 5, trial))
        if policy == 'none' and shots in {5, 10}:
            compare('more originals over 1 original', (strategy, policy, shots, trial), (strategy, 'none', 1, trial))
    return contrasts


def finetuning_tables(rows):
    """Build the exported tables once for both notebook display and figures."""
    metadata = ['strategy', 'shots', 'trial', 'selected_epoch', 'trainable_parameters', 'total_parameters', 'seconds']
    metrics = ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall']
    frame = pd.DataFrame([{**{name: row[name] for name in metadata},
                           'augmentation': row.get('augmentation', 'single'),
                           **{name: row['test'][name] for name in metrics}} for row in rows])
    frame['condition'] = frame['strategy'] + ' / ' + frame['augmentation']
    groups = ['strategy', 'augmentation', 'shots']
    summary = frame.groupby(groups).agg(
        accuracy_mean=('accuracy', 'mean'), accuracy_std=('accuracy', lambda x: np.std(x, ddof=0)),
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', lambda x: np.std(x, ddof=0))).reset_index()
    contrasts = pd.DataFrame(finetuning_contrasts(rows))
    contrast_summary = contrasts.groupby(['comparison', *groups])[
        ['accuracy_gain_pp', 'macro_f1_gain_pp']].agg(['mean', 'std', 'count'])
    recall = pd.DataFrame([{'strategy': row['strategy'], 'augmentation': row.get('augmentation', 'single'),
                            'shots': row['shots'], 'trial': row['trial'], 'class': name,
                            'recall': row['test']['class_recall'][label]}
                           for row in rows for label, name in enumerate(CLASSES)])
    return {'table': frame, 'summary': summary, 'contrasts': contrasts, 'contrast_summary': contrast_summary,
            'class_recall': recall,
            'class_recall_summary': recall.groupby([*groups, 'class'])['recall'].mean().unstack('class')}


def finetuning_figures(result):
    """Export shared tables and plot actual histories, paired scores and errors."""
    output, rows = Path(result['output']), result['trials']
    tables = finetuning_tables(rows)
    exports = {'table': 'trial-metrics', 'summary': 'summary', 'contrasts': 'paired-contrasts',
               'class_recall': 'class-recall'}
    for name, filename in exports.items():
        tables[name].to_csv(output / f'{filename}.csv', index=False)
    summary = tables['summary']
    title = 'SMOKE CHECK — ' if result['manifest']['config']['mode'] == 'smoke' else ''
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), layout='constrained')
        for (strategy, policy), group in summary.groupby(['strategy', 'augmentation']):
            for ax, metric in zip(axes, ['accuracy', 'macro_f1']):
                ax.errorbar(group['shots'], group[metric + '_mean'], yerr=group[metric + '_std'],
                            marker='o', capsize=3, label=f'{strategy} / {policy}')
                ax.set(xlabel='Labeled original images per class', ylabel=metric.replace('_', ' '),
                       ylim=(0, 1), xticks=sorted(summary.shots.unique()))
        axes[0].legend(frameon=False, fontsize=6)
        fig.suptitle(title + 'Matched neural-head control versus optional fine-tuning; mean ± trial SD')
        save_figure(fig, output, 'fine_tuning_comparison')
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), layout='constrained')
        maximum = max(r['shots'] for r in rows)
        for row in rows:
            if row['trial'] == 0 and row['shots'] == maximum:
                history = row['history']
                for ax, field in zip(axes, ['train_loss', 'dev_accuracy']):
                    label = row['strategy'] + ' / ' + row.get('augmentation', 'single')
                    ax.plot([h['epoch'] for h in history], [h[field] for h in history], marker='o', label=label)
                    ax.set(xlabel='Epoch', ylabel=field.replace('_', ' '))
        axes[0].legend(frameon=False, fontsize=6)
        fig.suptitle(title + f'Trial 0, {maximum}-shot; development selects epochs')
        save_figure(fig, output, 'fine_tuning_learning_curves')
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout='constrained')
        for ax, strategy in zip(axes, ['frozen_head', 'last_block']):
            row = next(r for r in rows if r['strategy'] == strategy and r['trial'] == 0
                       and r['shots'] == maximum and r.get('augmentation', 'none') == 'none')
            matrix = np.array(row['test']['confusion_matrix'])
            normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
            plot = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
            ax.set(xticks=range(len(CLASSES)), yticks=range(len(CLASSES)),
                   xticklabels=CLASSES, yticklabels=CLASSES,
                   xlabel='Predicted class', ylabel='True class', title=strategy)
            ax.tick_params(axis='x', labelrotation=90, labelsize=6)
            ax.tick_params(axis='y', labelsize=6)
        fig.colorbar(plot, ax=axes, label='Fraction within true class', shrink=.7)
        fig.suptitle(title + f'No augmentation, trial 0, {maximum}-shot; normalized confusion')
        save_figure(fig, output, 'fine_tuning_class_errors')
    return {**tables, 'figures': {name: str(output / f'{name}.png') for name in
            ['fine_tuning_comparison', 'fine_tuning_learning_curves', 'fine_tuning_class_errors']}}
