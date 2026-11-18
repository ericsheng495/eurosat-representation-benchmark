"""Read a run's results: per-episode metrics, summaries, and paired comparisons with 95% confidence intervals.

Every comparison within one shot budget is paired: the two sides used the same episodes.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import BASELINES, MODEL_IDS

KEYS = ['baseline', 'model', 'condition', 'shots']


def bootstrap_ci(values, resamples=2000, seed=42):
    """95% percentile bootstrap of the mean over episodes."""
    values = np.asarray(values, dtype=float)
    if len(values) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (resamples, len(values)))].mean(axis=1)
    return tuple(float(q) for q in np.quantile(means, [.025, .975]))


def metrics(confusion):
    """Accuracy, Macro-F1 and per-class recall for each episode from (episodes, 4, 4) confusion counts."""
    confusion = np.asarray(confusion, dtype=float)
    correct = np.diagonal(confusion, axis1=1, axis2=2)
    true, predicted = confusion.sum(axis=2), confusion.sum(axis=1)
    recall = np.divide(correct, true, out=np.zeros_like(correct), where=true > 0)
    precision = np.divide(correct, predicted, out=np.zeros_like(correct), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(correct), where=precision + recall > 0)
    return {'accuracy': correct.sum(1) / confusion.sum((1, 2)), 'macro_f1': f1.mean(1), 'recall': recall}


class Results:
    """One run. `groups` has a row per baseline x model x condition x shot; `episodes` a row per test episode."""

    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads(self.path.read_text())
        self.storage = storage = self.path.parents[2]  # <storage>/outputs/<run id>/results.json
        self.classes = self.meta['classes']
        self.groups = pd.DataFrame(self.meta['groups'])
        self.confusion, rows = {}, []
        for entry in self.meta['groups']:
            with np.load(storage / entry['file']) as saved:
                confusion = saved['confusion']
            group = tuple(entry[k] for k in KEYS)
            self.confusion[group] = confusion
            values = metrics(confusion)
            for episode in range(len(confusion)):
                row = {k: entry[k] for k in KEYS + ['track', 'control', 'rows']}
                row.update(episode=episode, accuracy=values['accuracy'][episode], macro_f1=values['macro_f1'][episode])
                row.update({f'recall_{name}': values['recall'][episode, i] for i, name in enumerate(self.classes)})
                rows.append(row)
        self.episodes = pd.DataFrame(rows)
        order = {'baseline': BASELINES, 'model': list(MODEL_IDS)}
        for column, values in order.items():
            for frame in [self.groups, self.episodes]:
                frame[column] = pd.Categorical(frame[column], [v for v in values if v in set(frame[column])], ordered=True)

    def summary(self, metric='accuracy'):
        """Mean and 95% CI of a metric per group, in percent."""
        rows = []
        for group, frame in self.episodes.groupby(KEYS, observed=True):
            low, high = bootstrap_ci(frame[metric] * 100)
            rows.append(dict(zip(KEYS, group), mean=frame[metric].mean() * 100, ci_low=low, ci_high=high,
                             episodes=len(frame)))
        return pd.DataFrame(rows)

    def paired(self, left, right, metric='accuracy'):
        """Mean paired difference (left - right) in percentage points, per remaining key.

        left and right are dicts that fix some keys, e.g. {'baseline': 'pretrained'} vs {'baseline': 'untrained'}.
        """
        varying = [k for k in KEYS if k not in left]
        a = self._select(left).set_index(varying + ['episode'])[metric]
        b = self._select(right).set_index(varying + ['episode'])[metric]
        joined = pd.concat([a.rename('left'), b.rename('right')], axis=1, join='inner').dropna()
        joined['difference'] = (joined['left'] - joined['right']) * 100
        rows = []
        for group, frame in joined.groupby(level=varying, observed=True):
            group = group if isinstance(group, tuple) else (group,)
            low, high = bootstrap_ci(frame['difference'])
            rows.append(dict(zip(varying, group), difference=frame['difference'].mean(), ci_low=low, ci_high=high,
                             share_not_better=float((frame['difference'] <= 0).mean()), episodes=len(frame)))
        return pd.DataFrame(rows)

    def contrast(self, weights, condition='none', metric='accuracy'):
        """Paired weighted contrast between models per baseline and shot, in percentage points.

        weights maps model to weight, e.g. {'vit_b_16': .5, 'dino_vits16': .5, 'resnet18': -.5, 'convnext_tiny': -.5}.
        Empty when a model in weights did not run.
        """
        frame = self.episodes[self.episodes.condition == condition]
        wide = frame.pivot_table(index=['baseline', 'shots', 'episode'], columns='model', values=metric, observed=True)
        if any(model not in wide.columns for model in weights):
            return pd.DataFrame()
        difference = sum(weight * wide[model] for model, weight in weights.items()) * 100
        rows = []
        for (baseline, shots), values in difference.groupby(level=['baseline', 'shots'], observed=True):
            low, high = bootstrap_ci(values)
            rows.append({'baseline': baseline, 'shots': shots, 'difference': values.mean(), 'ci_low': low, 'ci_high': high})
        return pd.DataFrame(rows)

    def _select(self, fixed):
        frame = self.episodes
        for column, value in fixed.items():
            frame = frame[frame[column] == value]
        return frame.drop(columns=list(fixed))


def load(path):
    return Results(path)
