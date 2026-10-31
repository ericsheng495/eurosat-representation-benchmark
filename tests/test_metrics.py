"""Metric definitions, cached confusion recovery and support-trial uncertainty."""
import unittest

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, f1_score,
                             matthews_corrcoef, precision_recall_fscore_support)

from eurosat.evaluation import complete_metrics, metrics_from_confusion, score, trial_interval


class MetricTests(unittest.TestCase):
    def test_multiclass_metrics_match_reference_on_imbalanced_classes(self):
        labels = np.repeat(np.arange(10), [40, 20, 15, 10, 5, 3, 2, 2, 2, 1])
        predictions = labels.copy()
        predictions[:30] = 1
        predictions[60:65] = 0
        result = score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, labels=np.arange(10), zero_division=0)
        np.testing.assert_allclose(result['class_precision'], precision)
        np.testing.assert_allclose(result['class_recall'], recall)
        np.testing.assert_allclose(result['class_f1'], f1)
        np.testing.assert_array_equal(result['class_support'], support)
        self.assertAlmostEqual(result['macro_precision'], precision.mean())
        self.assertAlmostEqual(result['macro_recall'], recall.mean())
        self.assertAlmostEqual(result['macro_f1'], f1.mean())
        self.assertAlmostEqual(result['weighted_f1'], f1_score(labels, predictions, average='weighted'))
        self.assertAlmostEqual(result['balanced_accuracy'], balanced_accuracy_score(labels, predictions))
        self.assertAlmostEqual(result['mcc'], matthews_corrcoef(labels, predictions))
        self.assertAlmostEqual(result['cohen_kappa'], cohen_kappa_score(labels, predictions))
        legacy = {'test': {'confusion_matrix': result['confusion_matrix']}}
        self.assertEqual(complete_metrics(legacy)['test'], result)

    def test_no_support_and_undefined_kappa_are_reported_explicitly(self):
        result = score([0, 0], [0, 0])
        self.assertEqual(result['balanced_accuracy'], 1.)
        self.assertEqual(result['macro_recall'], .1)
        self.assertEqual(result['class_precision'][1], 0.)
        self.assertIsNone(result['cohen_kappa'])
        self.assertEqual(result['mcc'], 0.)
        with self.assertRaises(ValueError):
            metrics_from_confusion(np.zeros((10, 10)))
        with self.assertRaises(ValueError):
            metrics_from_confusion(np.ones((10, 10)) * .5)

    def test_intervals_are_repeatable_and_need_more_than_smoke_trials(self):
        self.assertEqual(trial_interval([.5, .7]), (None, None))
        self.assertEqual(trial_interval([.4] * 20), (.4, .4))
        values = np.linspace(.4, .9, 20)
        first = trial_interval(values)
        self.assertEqual(first, trial_interval(values))
        self.assertLess(first[0], values.mean())
        self.assertGreater(first[1], values.mean())
