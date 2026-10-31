from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import nbformat
import numpy as np
from PIL import Image
import torch
from torch import nn

from eurosat import pipeline as p


class MeanFeatures(nn.Module):
    def forward(self, images):
        return images.mean((2, 3))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.old_hub = torch.hub.get_dir()
        self.addCleanup(torch.hub.set_dir, self.old_hub)
        self.storage = p.prepare_storage(self.root)

    def images(self, per_class=20):
        records = []
        for label, name in enumerate(p.CLASSES):
            (self.root / 'images' / name).mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                relative = f'{name}/{i}.jpg'
                Image.new('RGB', (24, 24), (label * 23, i * 7 % 256, (label + i) * 3)).save(
                    self.root / 'images' / relative)
                records.append({'path': relative, 'label': label, 'pixels_sha256': p.key([label, i])})
        return records

    def test_support_sets_are_nested_paired_and_disjoint_from_queries(self):
        records = self.images()
        split = p.stratified_split(records, 42)
        partitions = [set(indices) for indices in split.values()]
        self.assertEqual(len(set.union(*partitions)), len(records))
        self.assertFalse(partitions[0] & partitions[1] or partitions[0] & partitions[2]
                         or partitions[1] & partitions[2])
        config = p.Config(trials=3)
        trials = p.make_trials(records, split, config)
        self.assertEqual(trials, p.make_trials(records, split, config))
        one_shot_only = p.make_trials(records, split, replace(config, shots=[1], views=9))
        for full, reduced in zip(trials, one_shot_only):
            self.assertEqual(full['1'], reduced['1'])
        for trial in trials:
            self.assertLess(set(trial['1']), set(trial['5']))
            self.assertLess(set(trial['5']), set(trial['10']))
            self.assertTrue(set(trial['10']) <= partitions[0])

    def test_augmentation_is_reproducible_per_image_and_view(self):
        records = self.images(1)
        for policy in p.POLICIES:
            if policy == 'repeat':
                continue
            dataset = p.Images(self.root / 'images', records, [(0, 0), (0, 1)], policy, 42)
            first, label = dataset[0]
            self.assertEqual(label, 0)
            self.assertEqual(tuple(first.shape), (3, 224, 224))
            self.assertTrue(torch.equal(first, dataset[0][0]), policy)
            self.assertTrue(torch.isfinite(first).all())

    def test_pixel_duplicates_are_removed_before_splitting(self):
        records = self.images()
        archive = self.root / 'data/EuroSAT.zip'
        with zipfile.ZipFile(archive, 'w') as bundle:
            for record in records:
                bundle.write(self.root / 'images' / record['path'], '2750/' + record['path'])
            for name in p.CLASSES:
                bundle.write(self.root / 'images' / name / '0.jpg', f'2750/{name}/duplicate.jpg')
        with patch.object(p, 'DATA_MD5', p.digest_file(archive, 'md5')):
            _, kept, split, _ = p.prepare_data(self.storage, self.root / 'scratch', p.Config(per_class_limit=None))
        self.assertEqual(len(kept), 200)
        pixels = [r['pixels_sha256'] for r in kept]
        self.assertEqual(len(set(pixels)), 200)
        self.assertEqual(sum(map(len, split.values())), 200)

    def test_interrupted_feature_extraction_keeps_completed_chunks(self):
        records = self.images(1)
        config = p.Config(chunk_size=2, batch_size=2)
        class Interrupting(MeanFeatures):
            calls = 0
            fail = True
            def forward(self, images):
                self.calls += 1
                if self.fail and self.calls == 2:
                    raise RuntimeError('Disconnected')
                return super().forward(images)
        model = Interrupting()
        args = (self.storage, self.root / 'images', records, 'dataset', 'resnet18', model,
                torch.device('cpu'), [(i, 0) for i in range(5)], 'none', config)
        with self.assertRaisesRegex(RuntimeError, 'Disconnected'):
            p.extract(*args)
        self.assertEqual(len(list((self.root / 'cache/features').glob('*.npz'))), 1)
        model.calls, model.fail = 0, False
        features, labels = p.extract(*args)
        self.assertEqual(model.calls, 2)
        self.assertEqual(features.shape, (5, 3))
        self.assertEqual(len(labels), 5)
        np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1., atol=1e-6)
        model.calls = 0
        p.extract(*args)
        self.assertEqual(model.calls, 0)

    def test_trial_resume_and_completed_run_skip_model_loading(self):
        records = self.images()
        split = p.stratified_split(records, 42)
        config = p.Config(shots=[1, 5], policies=['none', 'repeat'], classifier_cs=[1.])
        fixture = (self.root / 'images', records, split, 'synthetic')
        actual_fit = p.fit_classifier
        calls = []
        def interrupted(*args):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError('Disconnected')
            return actual_fit(*args)
        with patch.object(p, 'check_device', return_value=torch.device('cpu')), \
                patch.object(p, 'prepare_data', return_value=fixture), \
                patch.object(p, 'load_backbone', return_value=MeanFeatures()):
            with patch.object(p, 'fit_classifier', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'Disconnected'):
                    p.run(config, self.root, self.root / 'scratch', 'local')
            with patch.object(p, 'fit_classifier', wraps=actual_fit) as fit:
                result = p.run(config, self.root, self.root / 'scratch', 'local')
                self.assertEqual(fit.call_count, 2)
            with patch.object(p, 'load_backbone', side_effect=AssertionError('Model should be skipped')):
                rerun = p.run(config, self.root, self.root / 'scratch', 'local')
        self.assertEqual(result['summary'], rerun['summary'])
        saved = json.loads((Path(result['output']) / 'results.json').read_text())
        self.assertEqual(len(saved['trials']), 4)
        for trial in saved['trials']:
            expected = trial['original_support_images'] * (2 if trial['policy'] == 'repeat' else 1)
            self.assertEqual(trial['classifier_rows'], expected)
        self.assertEqual(len(list((self.root / 'cache/experiments').rglob('*.npz'))), 4)

    def test_failed_atomic_write_preserves_previous_checkpoint(self):
        path = self.root / 'checkpoint.npz'
        path.write_bytes(b'previous')
        def fail(temporary):
            temporary.write_bytes(b'incomplete')
            raise RuntimeError('Interrupted')
        with self.assertRaises(RuntimeError):
            p.atomic(path, fail)
        self.assertEqual(path.read_bytes(), b'previous')

    def test_portable_notebook_is_valid_and_helpers_match(self):
        root = Path(__file__).resolve().parents[1]
        notebook = nbformat.read(root / 'Start_Here.ipynb', as_version=4)
        nbformat.validate(notebook)
        helpers = (root / 'src/eurosat/pipeline.py').read_text()
        self.assertTrue(any(c.source == helpers for c in notebook.cells))
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                compile(cell.source, 'notebook', 'exec')


if __name__ == '__main__':
    unittest.main()
