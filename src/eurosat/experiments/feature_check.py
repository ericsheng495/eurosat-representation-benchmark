"""Many-label check (table T5): do the embeddings still separate the four classes when labels are plentiful?

It uses the original-image embeddings the run already saved, so no model runs again.
  probe       the few-shot classifier, trained on every development image and scored on every test image.
              C is chosen by 5-fold cross-validation inside the development images; test labels choose nothing.
  dimensions  how many directions the embeddings really use: the participation ratio of their covariance,
              (sum of variances)^2 / (sum of squared variances), over all evaluated images.
  top share   variance in the top five directions. A six-class head needs exactly five.
"""
import numpy as np
from joblib import delayed
from sklearn.model_selection import StratifiedKFold

from ..config import BASE_CLASSES
from .classifier import _fit_predict

FOLDS = 5
TOP_DIRECTIONS = len(BASE_CLASSES) - 1


def _fold_accuracy(x, y, train, held, c, seed):
    pred, flag = _fit_predict(x[train], y[train], x[held], c, seed)
    return float(np.mean(pred == y[held])), flag


def feature_check(parallel, original, labels, split, cs, seed):
    """original = L2-normalized embeddings of every evaluated image; split = development / test image indices."""
    cs = sorted(cs)
    dev, test = np.array(split['dev']), np.array(split['test'])
    x, y = original[dev], labels[dev]
    folds = min(FOLDS, int(np.unique(y, return_counts=True)[1].min()))
    parts = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(x, y))
    scores = parallel(delayed(_fold_accuracy)(x, y, train, held, c, seed) for c in cs for train, held in parts)
    by_c = np.array([score for score, _ in scores]).reshape(len(cs), folds).mean(axis=1)
    best = int(np.argmax(by_c))  # First maximum = smallest C among ties.
    pred, flag = _fit_predict(x, y, original[test], cs[best], seed)
    variance = np.linalg.svd(original - original.mean(axis=0), compute_uv=False) ** 2
    return {'probe_accuracy': float(np.mean(pred == labels[test])), 'probe_c': cs[best],
            'probe_c_at_edge': best in (0, len(cs) - 1), 'cv_accuracy_by_c': by_c.tolist(), 'cs': cs,
            'folds': folds, 'train_images': len(dev), 'test_images': len(test),
            'nonconverged_fits': sum(f for _, f in scores) + flag,
            'effective_dimensions': float(variance.sum() ** 2 / (variance ** 2).sum()),
            'top_directions': TOP_DIRECTIONS, 'top_variance_share': float(variance[:TOP_DIRECTIONS].sum() / variance.sum()),
            'images': len(original)}
