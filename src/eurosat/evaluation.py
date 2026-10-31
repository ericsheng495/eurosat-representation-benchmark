"""Common classifier selection and paired held-out evaluation."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from .config import CLASSES


def score(labels, predictions):
    labels_all = list(range(len(CLASSES)))
    return {'accuracy': float(accuracy_score(labels, predictions)),
            'macro_f1': float(f1_score(labels, predictions, labels=labels_all, average='macro', zero_division=0)),
            'class_recall': recall_score(labels, predictions, labels=labels_all,
                                         average=None, zero_division=0).tolist(),
            'confusion_matrix': confusion_matrix(labels, predictions, labels=labels_all).tolist()}


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
    baseline = {(r['model'], r['shots'], r['trial']): r['test']['accuracy']
                for r in rows if r['policy'] == 'none'}
    groups = sorted({(r['model'], r['policy'], r['shots']) for r in rows})
    summary = []
    for model, policy, shots in groups:
        group = [r for r in rows if (r['model'], r['policy'], r['shots']) == (model, policy, shots)]
        scores = [r['test']['accuracy'] for r in group]
        gains = [r['test']['accuracy'] - baseline[model, shots, r['trial']] for r in group]
        summary.append({'model': model, 'policy': policy, 'shots': shots, 'trials': len(group),
                        'accuracy': float(np.mean(scores)), 'accuracy_std': float(np.std(scores)),
                        'macro_f1': float(np.mean([r['test']['macro_f1'] for r in group])),
                        'paired_gain_percentage_points': float(np.mean(gains) * 100),
                        'dev_accuracy': float(np.mean([r['dev_accuracy'] for r in group]))})
    return summary
