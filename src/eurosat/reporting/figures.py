"""The report figures F1-F11 (figure_n_table_sug.md), drawn from a run's results.

Each function saves one figure (PNG + PDF) and returns the PNG path. If the run lacks what a
figure needs (for example track 2 did not run), it raises MissingData with the reason.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from ..cache import key
from ..config import AUGMENTATION_METHODS, BASELINE_LABELS, BASELINES, MODEL_IDS, MODEL_LABELS
from ..experiments.conditions import sweep_views
from .style import (AXIS, INK, INK2, METHOD_COLORS, METHOD_LABELS, MODEL_COLORS, MODEL_SHORT, MUTED, REPEAT_COLOR,
                    STYLE, clean, colorbar, colormaps, header, heat, panel_title, save_figure, signed)

GAIN_ROWS = ['all', 'repeat-9', *AUGMENTATION_METHODS, 'repeat-2']
ROW_LABELS = {'all': 'All augmentations (track 2)', 'repeat-9': 'repeat ×9 (control)',
              'repeat-2': 'repeat ×2 (control)', **METHOD_LABELS}


class MissingData(Exception):
    """The run does not contain what this figure needs."""


def baselines_in(results):
    present = set(results.groups['baseline'].astype(str))
    return [b for b in BASELINES if b in present]


def models_in(results):
    present = set(results.groups['model'].astype(str))
    return [m for m in MODEL_IDS if m in present]


def shots_in(results):
    return sorted(set(results.groups['shots']))


def conditions_in(results):
    return set(results.groups['condition'])


def episodes_in(results):
    return int(results.groups['episodes'].max())


def gains(results, conditions, metric='accuracy'):
    """Paired gain over no augmentation (pp) per condition, baseline, model and shot."""
    frames = []
    for condition in conditions:
        frame = results.paired({'condition': condition}, {'condition': 'none'}, metric)
        frame.insert(0, 'condition', condition)
        frames.append(frame)
    return pd.concat(frames) if frames else pd.DataFrame()


def _error(frame, value='difference'):
    """Asymmetric error-bar lengths from the CI columns (zero when there is no CI)."""
    low = (frame[value] - frame['ci_low']).fillna(0).to_numpy()
    high = (frame['ci_high'] - frame[value]).fillna(0).to_numpy()
    return np.vstack([low, high])


def _legend_below(figure, handles, labels, columns):
    """Legend centered under the panels, clear of the title and the axes."""
    figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 0), ncol=columns, frameon=False,
                  fontsize=8.5, handletextpad=.5, columnspacing=1.8)


def _model_legend(figure, models):
    handles = [Line2D([], [], color=MODEL_COLORS[m], lw=2, marker='o', ms=6, mec='white', mew=1.2) for m in models]
    _legend_below(figure, handles, [MODEL_LABELS[m] for m in models], len(models))


def _forest(ax, groups, value_labels=False):
    """Grouped forest plot: a bold header per group, then one point with a 95% CI bar per row.

    groups = [(header, [(row label, one-row frame or None, color), ...]), ...]
    """
    side = blended_transform_factory(ax.transAxes, ax.transData)
    y, ticks, labels, heads = 0, [], [], []
    for title, rows in groups:
        heads.append(len(ticks)); ticks.append(y); labels.append(title); y += 1
        for label, row, color in rows:
            if row is not None and len(row):
                ax.errorbar(row.difference, [y], xerr=_error(row), fmt='o', color=color, ecolor=INK2,
                            elinewidth=2, ms=5.5, mec='white', mew=1.2, capsize=0)
                if value_labels:
                    r = row.iloc[0]
                    text = signed(r.difference) + ('' if r.ci_low != r.ci_low else f'  [{signed(r.ci_low)}, {signed(r.ci_high)}]')
                    ax.text(1.03, y, text, transform=side, fontsize=7, color=INK2, va='center')
            ticks.append(y); labels.append(label); y += 1
        y += .45
    ax.axvline(0, color=MUTED, lw=1.2)
    ax.set(yticks=ticks, ylim=(y - .55, -.7), xlabel='Paired accuracy difference (pp)')
    ax.set_yticklabels(labels)
    for i, label in enumerate(ax.get_yticklabels()):
        if i in heads:
            label.set_fontweight('bold'); label.set_color(INK)
    clean(ax, 'x')
    ax.spines['left'].set_visible(False)


def _panels(count, width=3.5, height=3.6, sharey=True):
    figure, axes = plt.subplots(1, count, figsize=(width * count + .4, height), sharey=sharey,
                                layout='constrained', squeeze=False)
    return figure, list(axes[0])


def augmentation_text(method, settings):
    """The setting of one method in words, e.g. 'σ 2.5–3.5 px'."""
    p = settings[method]
    if method == 'flip':
        return 'left-right or top-bottom'
    if method == 'rotation':
        return '90°, 180° or 270°'
    if method == 'scale_up':
        return f"center, ×{p['min_zoom']:g}–{p['max_zoom']:g}"
    if method == 'brightness_contrast':
        return (f"×{1 - p['max_change']:.1f}–{1 - p['min_change']:.1f} or "
                f"×{1 + p['min_change']:.1f}–{1 + p['max_change']:.1f}")
    if method == 'noise':
        return f"std {p['std']:g}"
    if method == 'blur':
        return f"σ {p['min_sigma']:g}–{p['max_sigma']:g} px"
    side = f"side {p['min_size_fraction'] * 100:.0f}–{p['max_size_fraction'] * 100:.0f}%"
    return f"{'gray' if method == 'erasing' else 'black'}, {side}"


def f1_augmented_views(prepared, config, output):
    """One development support image per class, the original and the view from each method used in the run."""
    from ..preprocessing.transforms import IMAGENET_MEAN, IMAGENET_STD, image_tensor
    if prepared is None:
        raise MissingData('needs the images: run the data check (Section 3) instead of report-only mode')
    supports = prepared.episodes['dev']['1'][0]['support']
    mean, std = np.array(IMAGENET_MEAN)[:, None, None], np.array(IMAGENET_STD)[:, None, None]
    columns = ['none', *AUGMENTATION_METHODS]
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(len(supports), len(columns), figsize=(1.25 * len(columns) + .6, 1.3 * len(supports) + .6),
                                    layout='constrained', squeeze=False)
        for r, index in enumerate(supports):
            record = prepared.records[index]
            for c, method in enumerate(columns):
                # The same seed as the experiment, so these are exactly the views the classifiers saw.
                seed = int(key([config.seed, record['pixels_sha256'], method])[:15], 16)
                tensor = image_tensor(prepared.root / record['path'], method, seed,
                                      config.augmentation.get(method), 0).numpy() * std + mean
                ax = axes[r, c]
                ax.imshow(np.clip(tensor.transpose(1, 2, 0), 0, 1))
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if r == 0:
                    title = 'original' if method == 'none' else METHOD_LABELS[method]
                    note = '64 → 224 px' if method == 'none' else augmentation_text(method, config.augmentation)
                    ax.set_title(f'{title}\n{note}', fontsize=7, fontweight='bold')
                if c == 0:
                    ax.set_ylabel(record['path'].split('/')[0].replace('Vegetation', '\nVegetation').replace('Crop', '\nCrop'),
                                  fontsize=7.5, color=INK2)
        header(figure, 'F1 · Augmented support views',
               'Development support images (episode 0, one per class); exactly the views used by the classifiers')
        return save_figure(figure, output, 'F1_augmented_views')


def f2_fewshot_curves(results, output):
    summary = results.summary()
    none = summary[summary.condition == 'none']
    models, shots = models_in(results), shots_in(results)
    with plt.rc_context(STYLE):
        figure, axes = _panels(len(baselines_in(results)))
        for ax, baseline in zip(axes, baselines_in(results)):
            for model in models:
                rows = none[(none.baseline == baseline) & (none.model == model)].sort_values('shots')
                if rows.ci_low.notna().all():
                    ax.fill_between(rows.shots, rows.ci_low, rows.ci_high, color=MODEL_COLORS[model], alpha=.13, lw=0)
                ax.plot(rows.shots, rows['mean'], color=MODEL_COLORS[model], lw=2, marker='o', ms=6, mec='white', mew=1.3)
            ax.axhline(25, color=MUTED, lw=1)
            ax.text(max(shots), 26.3, 'Chance (25%)', fontsize=7.5, color=MUTED, ha='right')
            ax.set(xticks=shots, ylim=(0, 100), xlabel='Labeled images per class (shots)')
            ax.set_xlim(min(shots) - .6, max(shots) + .6)
            clean(ax, 'y')
            panel_title(ax, BASELINE_LABELS[baseline])
        axes[0].set_ylabel('Test accuracy (%)')
        _model_legend(figure, models)
        header(figure, 'F2 · Few-shot accuracy by shot budget',
               f'No augmentation · mean of {episodes_in(results)} test episodes · band = 95% CI · 4-way, chance 25%')
        return save_figure(figure, output, 'F2_fewshot_curves')


def f3_model_differences(results, output):
    contrasts = [('ConvNeXt − ResNet', {'convnext_tiny': 1, 'resnet18': -1}),
                 ('DINO − ViT-B/16', {'dino_vits16': 1, 'vit_b_16': -1}),
                 ('Transformers − CNNs', {'vit_b_16': .5, 'dino_vits16': .5, 'resnet18': -.5, 'convnext_tiny': -.5})]
    data = [(title, results.contrast(weights)) for title, weights in contrasts]
    data = [(title, frame) for title, frame in data if len(frame)]
    if not data:
        raise MissingData('needs at least one model pair: ResNet-18 + ConvNeXt-Tiny, or ViT-B/16 + DINO ViT-S/16')
    shots = shots_in(results)
    with plt.rc_context(STYLE):
        figure, axes = _panels(len(baselines_in(results)), width=3.3, height=1.2 + .3 * len(data) * (len(shots) + 1.5))
        for ax, baseline in zip(axes, baselines_in(results)):
            _forest(ax, [(title, [(f'{shot}-shot', frame[(frame.baseline == baseline) & (frame.shots == shot)], INK)
                                  for shot in shots]) for title, frame in data])
            panel_title(ax, BASELINE_LABELS[baseline])
        header(figure, 'F3 · Paired model differences',
               'Same test episodes · point = mean difference · bar = 95% CI · right of 0 favors the first model')
        return save_figure(figure, output, 'F3_model_differences')


def f4_class_recall(results, output):
    sequential, _ = colormaps()
    columns = [f'recall_{name}' for name in results.classes]
    none = results.episodes[results.episodes.condition == 'none']
    means = none.groupby(['baseline', 'model', 'shots'], observed=True)[columns].mean() * 100
    models, shots, baselines = models_in(results), shots_in(results), baselines_in(results)
    short = [name.replace('HerbaceousVegetation', 'Herb.').replace('PermanentCrop', 'Perm.') for name in results.classes]
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(len(baselines), 1, figsize=(1.6 + .55 * len(shots) * len(columns), .9 + .5 * len(models) * len(baselines)),
                                    layout='constrained', squeeze=False)
        for ax, baseline in zip(axes[:, 0], baselines):
            values = np.array([[means.loc[(baseline, model, shot), col] if (baseline, model, shot) in means.index else np.nan
                                for shot in shots for col in columns] for model in models])
            image = heat(ax, values, sequential, Normalize(20, 100), lambda v: f'{v:.0f}', lambda v: v > 72, size=7)
            for x in np.arange(1, len(shots)) * len(columns) - .5:
                ax.axvline(x, color='white', lw=4)
            ax.set(yticks=range(len(models)), xticks=range(len(shots) * len(columns)))
            ax.set_yticklabels([MODEL_LABELS[m] for m in models], fontsize=8)
            ax.set_xticklabels(short * len(shots) if baseline == baselines[-1] else [], fontsize=7)
            ax.set_ylabel(BASELINE_LABELS[baseline], fontweight='bold', fontsize=9.5, color=INK, labelpad=8)
            if baseline == baselines[0]:
                for s, shot in enumerate(shots):
                    ax.text(s * len(columns) + (len(columns) - 1) / 2, -.7, f'{shot}-shot', ha='center', va='bottom',
                            fontsize=9, fontweight='bold')
        colorbar(figure, image, axes[:, 0], 'Recall (%)')
        header(figure, 'F4 · Per-class recall', 'No augmentation · mean recall over test episodes · darker = better')
        return save_figure(figure, output, 'F4_class_recall')


def f5_confusion(results, output):
    sequential, _ = colormaps()
    shot = max(shots_in(results))
    models, baselines = models_in(results), baselines_in(results)
    short = [name.replace('HerbaceousVegetation', 'Herb.').replace('PermanentCrop', 'Perm.') for name in results.classes]
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(len(models), len(baselines), figsize=(1.9 * len(baselines) + 1.6, 1.9 * len(models) + .6),
                                    layout='constrained', squeeze=False)
        for r, model in enumerate(models):
            for c, baseline in enumerate(baselines):
                ax = axes[r, c]
                counts = results.confusion.get((baseline, model, 'none', shot))
                if counts is None:
                    ax.axis('off')
                    continue
                total = counts.sum(0).astype(float)
                matrix = total / np.maximum(total.sum(1, keepdims=True), 1) * 100
                image = heat(ax, matrix, sequential, Normalize(0, 100), lambda v: f'{v:.0f}', lambda v: v > 55, size=6.5)
                ax.set(xticks=range(len(short)), yticks=range(len(short)))
                ax.set_xticklabels(short if r == len(models) - 1 else [], fontsize=6.5, rotation=30, ha='right',
                                   rotation_mode='anchor')
                ax.set_yticklabels(short if c == 0 else [], fontsize=6.5)
                if r == 0:
                    ax.set_title(BASELINE_LABELS[baseline], fontsize=9.5, fontweight='bold')
                if c == 0:
                    ax.set_ylabel(MODEL_LABELS[model], fontsize=8.5, fontweight='bold', color=INK)
        colorbar(figure, image, axes, 'Share of the true class (%)')
        header(figure, f'F5 · Confusion matrices at {shot} shots',
               'No augmentation · rows = true class, columns = predicted · rows sum to 100% · summed over test episodes')
        return save_figure(figure, output, 'F5_confusion')


def f11_baseline_effects(results, output):
    pairs = [(left, right, title) for left, right, title in
             [('pretrained', 'untrained', 'Pretrained − Untrained   (value of ImageNet pretraining)'),
              ('finetuned', 'pretrained', 'Fine-tuned − Pretrained   (value of fine-tuning)')]
             if {left, right} <= set(baselines_in(results))]
    if not pairs:
        raise MissingData('needs two baselines: untrained + pretrained, or pretrained + finetuned')
    models, shots = models_in(results), shots_in(results)
    with plt.rc_context(STYLE):
        figure, axes = _panels(len(pairs), width=4.6, height=1.2 + .3 * len(models) * (len(shots) + 1.4), sharey=False)
        for ax, (left, right, title) in zip(axes, pairs):
            frame = results.paired({'baseline': left, 'condition': 'none'}, {'baseline': right, 'condition': 'none'})
            _forest(ax, [(MODEL_LABELS[model], [(f'{shot}-shot', frame[(frame.model == model) & (frame.shots == shot)],
                                                 MODEL_COLORS[model]) for shot in shots]) for model in models],
                    value_labels=True)
            panel_title(ax, title)
        header(figure, 'F11 · Baseline effects',
               'What each training stage adds · same test episodes · bar = 95% CI · the panels have different scales')
        return save_figure(figure, output, 'F11_baseline_effects')


def f6_augmentation_gains(results, output):
    rows = [c for c in GAIN_ROWS if c in conditions_in(results)]
    if not rows:
        raise MissingData('needs track 2 or track 3')
    _, diverging = colormaps()
    frame = gains(results, rows)
    models, shots, baselines = models_in(results), shots_in(results), baselines_in(results)
    limit = max(3, np.ceil(np.nanpercentile(np.abs(frame.difference), 95)))
    with plt.rc_context(STYLE):
        figure, axes = _panels(len(baselines), width=.36 * len(models) * len(shots) + .4, height=.33 * len(rows) + 1.5)
        for ax, baseline in zip(axes, baselines):
            values = np.full((len(rows), len(models) * len(shots)), np.nan)
            bold = np.zeros_like(values, dtype=bool)
            for _, row in frame[frame.baseline == baseline].iterrows():
                r, c = rows.index(row.condition), models.index(str(row.model)) * len(shots) + shots.index(row.shots)
                values[r, c] = row.difference
                bold[r, c] = row.ci_low == row.ci_low and (row.ci_low > 0 or row.ci_high < 0)
            image = heat(ax, values, diverging, TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), signed,
                         lambda v: abs(v) > .62 * limit, bold=bold, size=6.3)
            for x in np.arange(1, len(models)) * len(shots) - .5:
                ax.axvline(x, color='white', lw=3.5)
            for r, condition in enumerate(rows[:-1]):
                if condition.startswith('repeat') or rows[r + 1] == 'repeat-2':
                    ax.axhline(r + .5, color='white', lw=3)
            ax.set(xticks=range(len(models) * len(shots)), yticks=range(len(rows)), xlabel='Shots per class')
            ax.set_xticklabels([str(s) for s in shots] * len(models), fontsize=7)
            ax.set_yticklabels([ROW_LABELS[c] for c in rows], fontsize=8)
            for m, model in enumerate(models):
                ax.text(m * len(shots) + (len(shots) - 1) / 2, -.75, MODEL_SHORT[model], ha='center', va='bottom',
                        fontsize=7.5, fontweight='bold')
            panel_title(ax, BASELINE_LABELS[baseline], pad=20)
        colorbar(figure, image, axes, 'Paired gain (pp)')
        header(figure, 'F6 · Augmentation gain over no augmentation',
               'Paired on the same test episodes · bold = 95% CI excludes 0 · repeat rows: as many rows, no new pixels')
        return save_figure(figure, output, 'F6_augmentation_gains')


def f7_all_gain_by_shot(results, output):
    if 'all' not in conditions_in(results):
        raise MissingData('needs track 2')
    frame = gains(results, ['all'])
    models, shots = models_in(results), shots_in(results)
    with plt.rc_context(STYLE):
        figure, axes = _panels(len(baselines_in(results)))
        for ax, baseline in zip(axes, baselines_in(results)):
            for i, model in enumerate(models):
                rows = frame[(frame.baseline == baseline) & (frame.model == model)].sort_values('shots')
                x = rows.shots.to_numpy(dtype=float) + (i - (len(models) - 1) / 2) * .16
                ax.errorbar(x, rows.difference, yerr=_error(rows), fmt='none', ecolor=MODEL_COLORS[model], elinewidth=1.5, alpha=.55)
                ax.plot(x, rows.difference, color=MODEL_COLORS[model], lw=2, marker='o', ms=6, mec='white', mew=1.2)
            ax.axhline(0, color=MUTED, lw=1)
            ax.set(xticks=shots, xlabel='Labeled images per class (shots)')
            ax.set_xlim(min(shots) - .6, max(shots) + .6)
            clean(ax, 'y')
            panel_title(ax, BASELINE_LABELS[baseline])
        axes[0].set_ylabel('Paired gain over no augmentation (pp)')
        _model_legend(figure, models)
        header(figure, 'F7 · All-augmentation gain by shot (track 2)',
               'Original + one view from each of the 8 methods · bar = 95% CI · 0 = no gain')
        return save_figure(figure, output, 'F7_all_gain_by_shot')


def f8_label_gap(results, output):
    shots = shots_in(results)
    if 'all' not in conditions_in(results) or not {1, 5} <= set(shots):
        raise MissingData('needs track 2 and the 1- and 5-shot budgets')
    summary = results.summary().set_index(['baseline', 'model', 'condition', 'shots'])['mean']
    models = models_in(results)
    lowest = summary.min() if len(summary) else 0
    with plt.rc_context(STYLE):
        figure, axes = _panels(len(baselines_in(results)), width=3.6, height=.55 * len(models) + 1.6)
        for ax, baseline in zip(axes, baselines_in(results)):
            for m, model in enumerate(models):
                point = lambda condition, shot: summary.get((baseline, model, condition, shot), np.nan)
                one, five, aug = point('none', 1), point('none', 5), point('all', 1)
                last = point('none', max(shots))
                ax.plot([one, last], [m, m], color=AXIS, lw=2, zorder=1)
                ax.plot([one, aug], [m, m], color=MODEL_COLORS[model], lw=5, solid_capstyle='round', zorder=2)
                ax.scatter(one, m, s=55, facecolor='white', edgecolor=INK2, lw=1.8, zorder=3)
                ax.scatter(aug, m, s=65, marker='D', color=MODEL_COLORS[model], edgecolor='white', lw=1.2, zorder=4)
                ax.scatter(five, m, s=55, color=INK, edgecolor='white', lw=1.2, zorder=3)
                if max(shots) > 5:
                    ax.scatter(last, m, s=55, marker='s', color=INK, edgecolor='white', lw=1.2, zorder=3)
                gap = five - one
                if gap == gap and abs(gap) > 1e-9:
                    ax.annotate(f'{(aug - one) / gap * 100:.0f}% closed', (max(last, five, aug), m), xytext=(8, 0),
                                textcoords='offset points', fontsize=7.5, color=INK2, va='center')
            ax.set(yticks=range(len(models)), ylim=(len(models) - .4, -.6), xlabel='Test accuracy (%)')
            ax.set_xlim(max(0, lowest - 5), 112)
            ax.set_yticklabels([MODEL_LABELS[m] for m in models])
            clean(ax, 'x')
            ax.spines['left'].set_visible(False)
            panel_title(ax, BASELINE_LABELS[baseline])
        handles = [Line2D([], [], ls='', marker='o', ms=7, mfc='white', mec=INK2, mew=1.6),
                   Line2D([], [], ls='', marker='D', ms=7, mfc=INK2, mec='white'),
                   Line2D([], [], ls='', marker='o', ms=7, mfc=INK, mec='white'),
                   Line2D([], [], ls='', marker='s', ms=7, mfc=INK, mec='white')]
        _legend_below(figure, handles, ['1-shot', '1-shot + all augmentations', '5-shot', f'{max(shots)}-shot'], 4)
        header(figure, 'F8 · How much of the label gap augmentation closes',
               'Means over different episodes (not paired) · closed = (◆ − ○) / (● − ○)')
        return save_figure(figure, output, 'F8_label_gap')


def f9_views(results, output):
    names = conditions_in(results)
    sweeps = {m: [(v, m if v == 1 else f'{m}-{v}') for v in sweep_views(m)] for m in AUGMENTATION_METHODS}
    sweeps = {m: [(v, n) for v, n in items if n in names] for m, items in sweeps.items()}
    sweeps = {m: items for m, items in sweeps.items() if len(items) > 1}
    if not sweeps or 1 not in shots_in(results):
        raise MissingData('needs the view sweep (VIEW_SWEEP = True)')
    repeats = sorted(int(n.split('-')[1]) for n in names if n.startswith('repeat-'))
    needed = sorted({n for items in sweeps.values() for _, n in items} | {f'repeat-{r}' for r in repeats})
    frame = gains(results, needed)
    frame = frame[frame.shots == 1].groupby(['baseline', 'condition'], observed=True)['difference'].mean()
    with plt.rc_context(STYLE):
        figure, axes = _panels(len(baselines_in(results)))
        for ax, baseline in zip(axes, baselines_in(results)):
            for method, items in sweeps.items():
                xs = [v for v, n in items if (baseline, n) in frame.index]
                ys = [frame[(baseline, n)] for v, n in items if (baseline, n) in frame.index]
                ax.plot(xs, ys, color=METHOD_COLORS[method], lw=2, marker='o', ms=5, mec='white', mew=1.1)
            xs = [r - 1 for r in repeats if (baseline, f'repeat-{r}') in frame.index]
            ax.plot(xs, [frame[(baseline, f'repeat-{x + 1}')] for x in xs], color=REPEAT_COLOR, lw=2, marker='o', ms=5,
                    mec='white', mew=1.1)
            ax.axhline(0, color=MUTED, lw=1)
            ax.set(xticks=sorted({v for items in sweeps.values() for v, _ in items}), xlabel='Augmented views per support image')
            clean(ax, 'y')
            panel_title(ax, BASELINE_LABELS[baseline])
        axes[0].set_ylabel('1-shot gain over no augmentation (pp)')
        handles = [Line2D([], [], color=METHOD_COLORS[m], lw=2, marker='o', ms=5, mec='white') for m in sweeps]
        handles.append(Line2D([], [], color=REPEAT_COLOR, lw=2, marker='o', ms=5, mec='white'))
        _legend_below(figure, handles, [METHOD_LABELS[m] for m in sweeps] + ['repeat (same rows, no new pixels)'], 5)
        header(figure, 'F9 · Effect of the number of augmented views',
               '1-shot · mean of the models · flip stops at 2 views and rotation at 3 (no more distinct views)')
        return save_figure(figure, output, 'F9_views')


def f10_class_effects(results, output):
    rows = [c for c in ['all', *AUGMENTATION_METHODS] if c in conditions_in(results)]
    if not rows or 1 not in shots_in(results):
        raise MissingData('needs track 2 or track 3 at 1 shot')
    _, diverging = colormaps()
    data = {}
    for name in results.classes:
        frame = gains(results, rows, metric=f'recall_{name}')
        data[name] = frame[frame.shots == 1].groupby(['baseline', 'condition'], observed=True)['difference'].mean()
    values_all = np.concatenate([series.to_numpy() for series in data.values()])
    limit = max(3, np.ceil(np.nanpercentile(np.abs(values_all), 95)))
    short = [name.replace('HerbaceousVegetation', 'Herbaceous\nvegetation').replace('PermanentCrop', 'Permanent\ncrop')
             for name in results.classes]
    with plt.rc_context(STYLE):
        figure, axes = _panels(len(baselines_in(results)), width=2.9, height=.33 * len(rows) + 1.5)
        for ax, baseline in zip(axes, baselines_in(results)):
            values = np.array([[data[name].get((baseline, row), np.nan) for name in results.classes] for row in rows])
            image = heat(ax, values, diverging, TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), signed,
                         lambda v: abs(v) > .62 * limit, size=7.5)
            if rows[0] == 'all' and len(rows) > 1:
                ax.axhline(.5, color='white', lw=4)
            ax.set(xticks=range(len(short)), yticks=range(len(rows)))
            ax.set_xticklabels(short, fontsize=7)
            ax.set_yticklabels([ROW_LABELS[r] for r in rows], fontsize=8)
            panel_title(ax, BASELINE_LABELS[baseline])
        colorbar(figure, image, axes, 'Recall change (pp)')
        header(figure, 'F10 · Class-level augmentation effects',
               '1-shot · paired recall change over no augmentation · mean of the models · single cells are noisy')
        return save_figure(figure, output, 'F10_class_effects')
