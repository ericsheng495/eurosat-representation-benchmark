"""Check paired report arithmetic and development-only policy selection."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from eurosat.config import Config, MODEL_IDS
from eurosat.evaluation import score
from eurosat.reporting import create_report


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        config = Config(models=['dinov2_vits14'], shots=[1], policies=['none', 'flip'], trials=2)
        rows = []
        for trial in range(2):
            for policy in config.policies:
                labels = np.tile(np.arange(10), 10)
                predictions = labels.copy()
                errors = 20 - trial * 10 - (10 if policy == 'flip' else 0)
                predictions[:errors] = (predictions[:errors] + 1) % 10
                rows.append({'model': 'dinov2_vits14', 'policy': policy, 'shots': 1, 'trial': trial,
                             'test': score(labels, predictions),
                             'dev_accuracy': .9 if policy == 'none' else .4,
                             'classifier_fit_seconds': .1, 'original_feature_seconds': 2.,
                             'parameter_count': 100})
        self.report = {'manifest': {'config': asdict(config), 'dataset': 'synthetic',
                                   'versions': {'fixture': True},
                                   'checkpoints': {'dinov2_vits14': MODEL_IDS['dinov2_vits14']}},
                       'trials': rows}
        self.source = self.root / 'results.json'
        self.source.write_text(json.dumps(self.report))

    def test_report_exports_real_figures_and_paired_gains_without_test_selection(self):
        import pandas as pd
        result = create_report(self.source)
        self.assertEqual(result['selected'][0]['policy'], 'none')
        rows = pd.read_csv(Path(result['output']) / 'paired-trials.csv')
        np.testing.assert_allclose(rows[rows.policy == 'flip'].gain_pp, 10.)
        for filename in result['figures'].values():
            with Image.open(filename) as image:
                image.verify()
            self.assertTrue(Path(filename).with_suffix('.pdf').read_bytes().startswith(b'%PDF'))
        saved = json.loads((Path(result['output']) / 'report.json').read_text())
        self.assertEqual(saved['mode'], 'smoke')
        self.assertIn('not a confidence interval', saved['variability'])

    def test_report_rejects_incomplete_pairs_before_writing_figures(self):
        self.report['trials'].pop()
        self.source.write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, 'complete, unique paired trials'):
            create_report(self.source)
        self.assertFalse((self.root / 'report').exists())
