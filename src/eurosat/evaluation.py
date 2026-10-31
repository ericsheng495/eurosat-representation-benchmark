"""Common classifier selection and paired held-out evaluation."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from .config import CLASSES


MIN_CI_TRIALS = 5
EXTRA_METRICS = ['macro_precision', 'macro_recall', 'balanced_accuracy',
                 'weighted_f1', 'mcc', 'cohen_kappa']


def metrics_from_confusion(matrix):
    """Compute class metrics directly from counts, including old cached results."""
    cm = np.asarray(matrix, dtype=float)
    if cm.shape != (len(CLASSES), len(CLASSES)) or not np.isfinite(cm).all() or np.any(cm < 0) or np.any(cm != np.floor(cm)):
        raise ValueError('Expected a finite nonnegative ten-class count matrix')
    total = cm.sum()
    if total == 0:
        raise ValueError('Cannot score an empty query set')
    support, predicted, correct = cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
    divide = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    precision, recall = divide(correct, predicted), divide(correct, support)
    f1 = divide(2 * precision * recall, precision + recall)
    accuracy = correct.sum() / total
    expected = np.dot(support, predicted) / total ** 2
    denominator = np.sqrt((total ** 2 - np.dot(support, support)) *
                          (total ** 2 - np.dot(predicted, predicted)))
    return {'accuracy': float(accuracy), 'macro_f1': float(f1.mean()),
            'macro_precision': float(precision.mean()), 'macro_recall': float(recall.mean()),
            'weighted_f1': float(np.dot(f1, support) / total),
            'balanced_accuracy': float(recall[support > 0].mean()),
            'mcc': float((correct.sum() * total - np.dot(support, predicted)) / denominator) if denominator else 0.,
            'cohen_kappa': float((accuracy - expected) / (1 - expected)) if expected < 1 else None,
            'class_precision': precision.tolist(), 'class_recall': recall.tolist(),
            'class_f1': f1.tolist(), 'class_support': support.astype(int).tolist(),
            'confusion_matrix': cm.astype(int).tolist()}


def score(labels, predictions):
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(CLASSES))))
    return metrics_from_confusion(matrix)


def complete_metrics(row):
    """Upgrade confusion-derived metrics without inference or classifier refitting."""
    if 'confusion_matrix' in row['test']:
        row['test'].update(metrics_from_confusion(row['test']['confusion_matrix']))
    return row


def trial_interval(values, *, seed=42, resamples=2000):
    """95% percentile bootstrap over support trials, conditional on fixed queries."""
    values = np.asarray(values, dtype=float)
    if len(values) < MIN_CI_TRIALS:
        return None, None
    if np.all(values == values[0]):
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    return float(low), float(high)


def fit_classifier(x, y, dev_x, dev_y, config):
    best, selected, dev_accuracy = None, None, -1.
    for c in sorted(config.classifier_cs):
        # lbfgs uses multinomial loss for this ten-class task and L2 regularization.
        classifier = LogisticRegression(C=c, solver='lbfgs', max_iter=2000,
                                        random_state=config.seed)
        classifier.fit(x, y)
        accuracy = classifier.score(dev_x, dev_y)
        if accuracy > dev_accuracy:
            best, selected, dev_accuracy = classifier, c, float(accuracy)
    return best, selected, dev_accuracy


def summarize(rows):
    rows = [complete_metrics(r) for r in rows]
    baseline = {(r['model'], r['shots'], r['trial']): r['test']['accuracy']
                for r in rows if r['policy'] == 'none'}
    groups = sorted({(r['model'], r['policy'], r['shots']) for r in rows})
    summary = []
    for model, policy, shots in groups:
        group = sorted([r for r in rows if (r['model'], r['policy'], r['shots']) == (model, policy, shots)],
                       key=lambda r: r['trial'])
        scores = [r['test']['accuracy'] for r in group]
        gains = [r['test']['accuracy'] - baseline[model, shots, r['trial']] for r in group]
        summary.append({'model': model, 'policy': policy, 'shots': shots, 'trials': len(group),
                        'accuracy': float(np.mean(scores)), 'accuracy_std': float(np.std(scores)),
                        'macro_f1': float(np.mean([r['test']['macro_f1'] for r in group])),
                        'macro_f1_std': float(np.std([r['test']['macro_f1'] for r in group])),
                        'paired_gain_std_percentage_points': float(np.std(gains) * 100),
                        'fraction_nonpositive_gain': float(np.mean(np.array(gains) <= 0)),
                        'paired_gain_percentage_points': float(np.mean(gains) * 100),
                        'dev_accuracy': float(np.mean([r['dev_accuracy'] for r in group]))})
        record = summary[-1]
        for metric in EXTRA_METRICS:
            values = [r['test'][metric] for r in group if r['test'].get(metric) is not None]
            record[metric] = float(np.mean(values)) if len(values) == len(group) else None
            record[metric + '_std'] = float(np.std(values)) if len(values) == len(group) else None
        for metric, values in [('accuracy', scores), ('macro_f1', [r['test']['macro_f1'] for r in group]),
                               ('paired_gain_percentage_points', np.array(gains) * 100)]:
            record[metric + '_ci95_low'], record[metric + '_ci95_high'] = trial_interval(values)
    return summary
