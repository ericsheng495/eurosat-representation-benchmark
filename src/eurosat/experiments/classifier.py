"""The shared few-shot classifier: L2-regularized multinomial logistic regression on L2-normalized embeddings.

For each condition and shot budget:
  1. choose C on the development episodes (mean accuracy; ties keep the smaller C)
  2. with that C fixed, fit a new classifier on each test episode's supports and predict its queries
Test labels never choose anything. Fits run in parallel across episodes.
"""
import time
import warnings

import numpy as np
from joblib import delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

MAX_ITER = 5000


def support_rows(original, views, labels, support, condition):
    """Training rows for one episode: the originals (copies times) plus their augmented views."""
    x = [np.tile(original[support], (condition.copies, 1))]
    y = [np.tile(labels[support], condition.copies)]
    for item in condition.views:
        table, row_of = views[item]
        x.append(table[[row_of[i] for i in support]])
        y.append(labels[support])
    return np.concatenate(x), np.concatenate(y)


def _fit_predict(x, y, query_x, c, seed):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        model = LogisticRegression(C=c, max_iter=MAX_ITER, random_state=seed).fit(x, y)
    return model.predict(query_x), int(np.max(model.n_iter_) >= MAX_ITER)


def _development_episode(x, y, query_x, query_y, cs, seed):
    """Accuracy for every C on one development episode."""
    results = [_fit_predict(x, y, query_x, c, seed) for c in cs]
    return [float(np.mean(pred == query_y)) for pred, _ in results], sum(flag for _, flag in results)


def _test_episode(x, y, query_x, query_y, c, seed, label_order):
    """4x4 confusion counts (rows = true class) for one test episode."""
    pred, flag = _fit_predict(x, y, query_x, c, seed)
    index = {label: i for i, label in enumerate(label_order)}
    matrix = np.zeros((len(label_order), len(label_order)), dtype=np.int32)
    np.add.at(matrix, ([index[v] for v in query_y], [index[v] for v in pred]), 1)
    return matrix, flag


def evaluate_condition(parallel, original, views, labels, label_order, episodes, condition, shot, cs, seed):
    """Choose C on development episodes, then score every test episode. Returns arrays for one result group."""
    cs = sorted(cs)
    started = time.perf_counter()

    def rows(episode):
        x, y = support_rows(original, views, labels, episode['support'], condition)
        return x, y, original[episode['query']], labels[episode['query']]

    development = parallel(delayed(_development_episode)(*rows(ep), cs, seed) for ep in episodes['dev'][str(shot)])
    dev_accuracy = np.mean([accuracies for accuracies, _ in development], axis=0)
    best = int(np.argmax(dev_accuracy))  # First maximum = smallest C among ties.
    test = parallel(delayed(_test_episode)(*rows(ep), cs[best], seed, label_order)
                    for ep in episodes['test'][str(shot)])
    return {
        'confusion': np.stack([matrix for matrix, _ in test]),
        'dev_accuracy_by_c': dev_accuracy, 'cs': np.array(cs), 'c': cs[best],
        'c_at_edge': best in (0, len(cs) - 1),
        'nonconverged_fits': sum(flag for _, flag in development) + sum(flag for _, flag in test),
        'fits': len(development) * len(cs) + len(test), 'fit_seconds': time.perf_counter() - started,
    }
