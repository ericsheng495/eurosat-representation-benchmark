"""Combine compatible teammate exports without mixing experimental protocols."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path

from .cache import atomic, digest_file, key, write_json
from .config import CLASSES, MODEL_IDS
from .evaluation import complete_metrics, summarize, trial_interval
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
    manifest, rows = report['manifest'], [complete_metrics(r) for r in report['trials']]
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
        'macro_precision': r['test'].get('macro_precision'), 'macro_recall': r['test'].get('macro_recall'),
        'balanced_accuracy': r['test'].get('balanced_accuracy'), 'weighted_f1': r['test'].get('weighted_f1'),
        'mcc': r['test'].get('mcc'), 'cohen_kappa': r['test'].get('cohen_kappa'),
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
    names.extend(proposal_analysis(report, output))
    write_json(output / 'report.json', {'mode': config['mode'], 'trials': len(rows),
               'models': config['models'], 'figures': names,
               'uncertainty': {'confidence_level': .95, 'method': 'percentile support-trial bootstrap',
                               'resamples': 2000, 'seed': 42, 'minimum_trials': 5,
                               'scope': 'Conditional on fixed query images; no spatial/generalization or multiplicity coverage'},
               'variability': 'Population standard deviation across paired trials; not a confidence interval'})
    print('Result figures and tables:', output, flush=True)
    return {'output': str(output), 'figures': {name: str(output / f'{name}.png') for name in names},
            'selected': selected}


def compare_budgets(primary_file, nine_file, output):
    """Compare paired original-label and view budgets from the proposal."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    primary = json.loads(Path(primary_file).read_text())
    nine = json.loads(Path(nine_file).read_text())
    first, second = primary['manifest'], nine['manifest']
    def common(manifest):
        config = deepcopy(manifest['config'])
        for field in ['shots', 'views', 'policies']:
            config.pop(field)
        return {'config': config, 'dataset': manifest['dataset'],
                'checkpoints': manifest['checkpoints'], 'versions': manifest['versions']}
    if common(first) != common(second) or first['config']['views'] != 4 or second['config']['views'] != 9:
        raise ValueError('Budget comparisons need the same protocol and four-/nine-view runs')
    if set(first['config']['shots']) != {1, 5, 10} or second['config']['shots'] != [1]:
        raise ValueError('Budget comparisons need primary 1/5/10 shots and follow-up 1-shot')
    if [t['1'] for t in first['trials']] != [t['1'] for t in second['trials']]:
        raise ValueError('Original 1-shot support images must be identical across view budgets')
    for report in [primary, nine]:
        config = report['manifest']['config']
        expected = {(m, p, k, t) for m in config['models'] for p in config['policies']
                    for k in config['shots'] for t in range(config['trials'])}
        actual = [(r['model'], r['policy'], r['shots'], r['trial']) for r in report['trials']]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError('Budget comparisons need complete, unique paired trials')
    lookup = lambda report: {(r['model'], r['policy'], r['shots'], r['trial']): r for r in report['trials']}
    four, extra = lookup(primary), lookup(nine)
    comparisons = []
    for model in first['config']['models']:
        for trial in range(first['config']['trials']):
            conditions = []
            for policy in first['config']['policies']:
                if policy != 'none':
                    conditions.append(('one_plus_four_vs_five_originals', policy,
                                       four[model, policy, 1, trial], four[model, 'none', 5, trial]))
            for policy in second['config']['policies']:
                if policy != 'none':
                    if policy not in first['config']['policies']:
                        raise ValueError('Nine-view methods must also appear in the primary grid')
                    conditions.extend([
                        ('one_plus_nine_vs_ten_originals', policy, extra[model, policy, 1, trial],
                         four[model, 'none', 10, trial]),
                        ('nine_vs_four_views', policy, extra[model, policy, 1, trial], four[model, policy, 1, trial])])
            for name, policy, augmented, reference in conditions:
                comparisons.append({'comparison': name, 'model': model, 'policy': policy, 'trial': trial,
                                    'accuracy': augmented['test']['accuracy'],
                                    'reference_accuracy': reference['test']['accuracy'],
                                    'difference_percentage_points': 100 * (augmented['test']['accuracy'] - reference['test']['accuracy']),
                                    'original_images': augmented['original_support_images'],
                                    'reference_original_images': reference['original_support_images'],
                                    'classifier_rows': augmented['classifier_rows'],
                                    'reference_classifier_rows': reference['classifier_rows']})
    frame = pd.DataFrame(comparisons)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic(output / 'budget-comparisons.csv', lambda p: frame.to_csv(p, index=False))
    summary = frame.groupby(['comparison', 'model', 'policy']).difference_percentage_points.agg(
        mean='mean', std=lambda x: np.std(x, ddof=0), fraction_negative=lambda x: np.mean(x < 0)).reset_index()
    intervals = frame.groupby(['comparison', 'model', 'policy']).difference_percentage_points.apply(
        lambda values: pd.Series(dict(zip(['ci95_low', 'ci95_high'], trial_interval(values)))))
    summary = summary.merge(intervals.unstack().reset_index(), on=['comparison', 'model', 'policy'])
    atomic(output / 'budget-summary.csv', lambda p: summary.to_csv(p, index=False))
    smoke = first['config']['mode'] == 'smoke'
    label = 'SMOKE CHECK — execution validation' if smoke else 'DINOv2 view-budget comparisons'
    kinds = [('one_plus_four_vs_five_originals', '1 original + 4 views versus 5 originals'),
             ('one_plus_nine_vs_ten_originals', '1 original + 9 views versus 10 originals'),
             ('nine_vs_four_views', '9 versus 4 added views at 1-shot')]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
        for ax, (kind, title) in zip(axes, kinds):
            data = summary[summary.comparison == kind]
            labels = (data.model + ' / ' if len(first['config']['models']) > 1 else '') + data.policy.str.replace('_', ' / ', regex=False)
            ax.barh(labels, data['mean'], xerr=data['std'], color='#376D95', capsize=2)
            ax.axvline(0, color='#687682', linewidth=.8)
            ax.set(title=title, xlabel='Paired difference (percentage points)')
            ax.invert_yaxis()
        fig.suptitle(label + '\nMean ± paired-trial SD; augmented views are not new labeled originals')
        save_figure(fig, output, 'label_and_view_budgets')
    result = {'mode': first['config']['mode'], 'output': str(output), 'comparisons': len(frame),
              'figures': {'label_and_view_budgets': str(output / 'label_and_view_budgets.png')},
              'sources': {'four_views': digest_file(primary_file), 'nine_views': digest_file(nine_file)}}
    write_json(output / 'comparisons.json', result)
    return result


def proposal_analysis(report, output):
    """Additional proposal outcomes: Macro-F1, per-class gains, costs and negatives."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    rows = report['trials']
    config = report['manifest']['config']
    baselines = {(r['model'], r['shots'], r['trial']): r for r in rows if r['policy'] == 'none'}
    class_rows = []
    for row in rows:
        baseline = baselines[row['model'], row['shots'], row['trial']]
        for label, name in enumerate(CLASSES):
            class_rows.append({'model': row['model'], 'shots': row['shots'], 'policy': row['policy'],
                               'trial': row['trial'], 'class': name, 'recall': row['test']['class_recall'][label],
                               'precision': row['test'].get('class_precision', [None] * len(CLASSES))[label],
                               'f1': row['test'].get('class_f1', [None] * len(CLASSES))[label],
                               'support': row['test'].get('class_support', [None] * len(CLASSES))[label],
                               'recall_gain_pp': 100 * (row['test']['class_recall'][label] - baseline['test']['class_recall'][label])})
    classes = pd.DataFrame(class_rows)
    atomic(output / 'class-effects.csv', lambda p: classes.to_csv(p, index=False))
    classification = classes.groupby(['model', 'policy', 'shots', 'class']).agg(
        precision=('precision', 'mean'), recall=('recall', 'mean'), f1=('f1', 'mean'),
        precision_std=('precision', lambda x: np.std(x, ddof=0)),
        recall_std=('recall', lambda x: np.std(x, ddof=0)), f1_std=('f1', lambda x: np.std(x, ddof=0)),
        query_images_per_trial=('support', 'first')).reset_index()
    atomic(output / 'classification-report.csv', lambda p: classification.to_csv(p, index=False))
    stage_rows = []
    for execution in report.get('executions', []):
        for stage in execution.get('stages', []):
            stage_rows.append({'utc': execution['utc'], 'device': execution['device'],
                               'hardware': execution['hardware'], **stage})
    atomic(output / 'cost-stages.csv', lambda p: pd.DataFrame(stage_rows).to_csv(p, index=False))
    contrasts = []
    models = config['models']
    indexed = {(r['model'], r['policy'], r['shots'], r['trial']): r for r in rows}
    for i, first in enumerate(models):
        for second in models[i + 1:]:
            for policy in config['policies']:
                for shot in config['shots']:
                    for trial in range(config['trials']):
                        a, b = indexed[first, policy, shot, trial], indexed[second, policy, shot, trial]
                        contrasts.append({'model': first, 'reference': second, 'policy': policy, 'shots': shot,
                                          'trial': trial, 'accuracy_difference_pp': 100 * (a['test']['accuracy'] - b['test']['accuracy']),
                                          'augmentation_gain_difference_pp': 100 * (
                                              a['test']['accuracy'] - baselines[first, shot, trial]['test']['accuracy']
                                              - b['test']['accuracy'] + baselines[second, shot, trial]['test']['accuracy'])})
    if contrasts:
        comparison_table = pd.DataFrame(contrasts)
        atomic(output / 'model-comparisons.csv', lambda p: comparison_table.to_csv(p, index=False))
        differences = comparison_table.groupby(['model', 'reference', 'policy', 'shots'])[
            ['accuracy_difference_pp', 'augmentation_gain_difference_pp']].agg(['mean', lambda x: np.std(x, ddof=0)])
        differences.columns = ['_'.join(column).replace('<lambda_0>', 'std') for column in differences.columns]
        differences = differences.reset_index()
        intervals = []
        for indices, group in comparison_table.groupby(['model', 'reference', 'policy', 'shots']):
            record = dict(zip(['model', 'reference', 'policy', 'shots'], indices))
            for metric in ['accuracy_difference_pp', 'augmentation_gain_difference_pp']:
                record[metric + '_ci95_low'], record[metric + '_ci95_high'] = trial_interval(
                    group.sort_values('trial')[metric])
            intervals.append(record)
        differences = differences.merge(pd.DataFrame(intervals), on=['model', 'reference', 'policy', 'shots'])
        atomic(output / 'model-comparison-summary.csv', lambda p: differences.to_csv(p, index=False))
    summary = pd.DataFrame(summarize(rows))
    negatives = summary[(summary.policy != 'none') & (summary.paired_gain_percentage_points <= 0)]
    atomic(output / 'nonpositive-effects.csv', lambda p: negatives.to_csv(p, index=False))
    finding = ('SMOKE CHECK: these results validate execution only.\n\n' if config['mode'] == 'smoke' else '')
    finding += (f'{len(negatives)} model/policy/shot conditions had a nonpositive mean paired gain. '
                'All conditions, including negative and inconsistent outcomes, remain in summary.csv.\n\n'
                'Selection used development labels in addition to classifier-shot labels. '
                'Augmented views are not independent labeled originals. Image-level splitting does not establish geographic independence. '
                'DINOv2/ViT differences do not isolate self-supervision from model size and pretraining data.\n\n'
                'Cost stages distinguish cache hits and newly computed chunks. Process RSS is a lifetime peak; '
                'CUDA memory is a per-stage allocated peak; MPS memory is an end-of-stage allocation snapshot. '
                'These measures are not directly interchangeable. Model loading is recorded separately.\n')
    atomic(output / 'interpretation.md', lambda p: p.write_text(finding))
    label = 'SMOKE CHECK — execution validation' if config['mode'] == 'smoke' else 'Frozen representations'
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 3.6), layout='constrained')
        for model in models:
            group = [r for r in rows if r['model'] == model and r['policy'] == 'none']
            shots = sorted(config['shots'])
            values = [[r['test']['macro_f1'] for r in group if r['shots'] == shot] for shot in shots]
            ax.errorbar(shots, [np.mean(v) for v in values], yerr=[np.std(v) for v in values],
                        marker='o', capsize=3, label=model)
        ax.set(xlabel='Original images per class', ylabel='Macro-F1', ylim=(0, 1), xticks=sorted(config['shots']),
               title=label + '\nUnaugmented baseline; mean ± trial SD')
        ax.legend(frameon=False)
        save_figure(fig, output, 'macro_f1_by_shot')
        fig, axes = plt.subplots(len(models), len(config['shots']), squeeze=False,
                                 figsize=(5 * len(config['shots']), 4.5 * len(models)), layout='constrained')
        for i, model in enumerate(models):
            for j, shots in enumerate(sorted(config['shots'])):
                frame = classes[(classes.model == model) & (classes.shots == shots) & (classes.policy != 'none')]
                values = frame.pivot_table(index='policy', columns='class', values='recall_gain_pp', aggfunc='mean')
                if values.empty:
                    axes[i, j].set_axis_off()
                    continue
                sns.heatmap(values.reindex(columns=CLASSES), cmap='RdBu', center=0, vmin=-100, vmax=100,
                            ax=axes[i, j], cbar=True)
                axes[i, j].set(title=f'{model} · {shots}-shot', xlabel=None, ylabel=None)
                axes[i, j].tick_params(axis='x', labelrotation=90, labelsize=7)
        fig.suptitle(label + '\nClass recall change versus paired baseline (percentage points)')
        save_figure(fig, output, 'class_augmentation_deltas')
    extra = metric_overview(report, output)
    return ['macro_f1_by_shot', 'class_augmentation_deltas', *extra]


def metric_overview(report, output):
    """Compare classification metrics against the fixed no-augmentation baseline."""
    import matplotlib.pyplot as plt
    summary = summarize(report['trials'])
    config = report['manifest']['config']
    metrics = [('accuracy', 'Accuracy'), ('macro_precision', 'Macro precision'),
               ('macro_recall', 'Macro recall'), ('macro_f1', 'Macro-F1'),
               ('weighted_f1', 'Weighted F1'), ('balanced_accuracy', 'Balanced accuracy')]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(10, 5.8), layout='constrained')
        for ax, (metric, title) in zip(axes.flat, metrics):
            for model in config['models']:
                rows = sorted([r for r in summary if r['model'] == model and r['policy'] == 'none'],
                              key=lambda r: r['shots'])
                if any(r.get(metric) is None for r in rows):
                    continue
                ax.errorbar([r['shots'] for r in rows], [r[metric] for r in rows],
                            yerr=[r.get(metric + '_std', 0.) for r in rows],
                            marker='o', capsize=2, label=model)
            ax.set(title=title, xlabel='Original images per class', ylim=(0, 1), xticks=sorted(config['shots']))
        axes[0, 0].legend(frameon=False)
        fig.suptitle(('SMOKE CHECK — ' if config['mode'] == 'smoke' else '') +
                     'No-augmentation model baselines; mean ± trial SD')
        save_figure(fig, output, 'classification_metrics')
    return ['classification_metrics']
