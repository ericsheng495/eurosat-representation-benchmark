from dataclasses import replace, asdict
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
from eurosat import cache, config as settings, data, features, evaluation


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
        self.storage = cache.prepare_storage(self.root)

    def images(self, per_class=20):
        records = []
        for label, name in enumerate(settings.CLASSES):
            (self.root / 'images' / name).mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                relative = f'{name}/{i}.jpg'
                Image.new('RGB', (24, 24), (label * 23, i * 7 % 256, (label + i) * 3)).save(
                    self.root / 'images' / relative)
                records.append({'path': relative, 'label': label, 'pixels_sha256': cache.key([label, i])})
        return records

    def test_support_sets_are_nested_paired_and_disjoint_from_queries(self):
        records = self.images()
        split = data.stratified_split(records, 42)
        partitions = [set(indices) for indices in split.values()]
        self.assertEqual(len(set.union(*partitions)), len(records))
        self.assertFalse(partitions[0] & partitions[1] or partitions[0] & partitions[2]
                         or partitions[1] & partitions[2])
        config = settings.Config(trials=3)
        trials = data.make_trials(records, split, config)
        self.assertEqual(trials, data.make_trials(records, split, config))
        one_shot_only = data.make_trials(records, split, replace(config, shots=[1], views=9))
        for full, reduced in zip(trials, one_shot_only):
            self.assertEqual(full['1'], reduced['1'])
        for trial in trials:
            self.assertLess(set(trial['1']), set(trial['5']))
            self.assertLess(set(trial['5']), set(trial['10']))
            self.assertTrue(set(trial['10']) <= partitions[0])

    def test_augmentation_is_reproducible_per_image_and_view(self):
        records = self.images(1)
        for policy in settings.POLICIES + settings.OPTIONAL_POLICIES:
            if policy == 'repeat':
                continue
            dataset = features.Images(self.root / 'images', records, [(0, 0), (0, 1)], policy, 42)
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
            for name in settings.CLASSES:
                bundle.write(self.root / 'images' / name / '0.jpg', f'2750/{name}/duplicate.jpg')
        with patch.object(data, 'DATA_MD5', cache.digest_file(archive, 'md5')):
            _, kept, split, _ = data.prepare_data(self.storage, self.root / 'scratch', settings.Config(per_class_limit=None))
        self.assertEqual(len(kept), 200)
        pixels = [r['pixels_sha256'] for r in kept]
        self.assertEqual(len(set(pixels)), 200)
        self.assertEqual(sum(map(len, split.values())), 200)

    def test_interrupted_feature_extraction_keeps_completed_chunks(self):
        records = self.images(1)
        config = settings.Config(chunk_size=2, batch_size=2)
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
            features.extract(*args)
        self.assertEqual(len(list((self.root / 'cache/features').glob('*.npz'))), 1)
        model.calls, model.fail = 0, False
        embeddings, labels = features.extract(*args)
        self.assertEqual(model.calls, 2)
        self.assertEqual(embeddings.shape, (5, 3))
        self.assertEqual(len(labels), 5)
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1., atol=1e-6)
        model.calls = 0
        features.extract(*args)
        self.assertEqual(model.calls, 0)

    def test_trial_resume_and_completed_run_skip_model_loading(self):
        records = self.images()
        split = data.stratified_split(records, 42)
        config = settings.Config(shots=[1, 5], policies=['none', 'repeat'], classifier_cs=[1.])
        fixture = (self.root / 'images', records, split, 'synthetic')
        actual_fit = evaluation.fit_classifier
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

    def test_new_metrics_upgrade_cached_trials_without_models_or_refitting(self):
        records = self.images()
        split = data.stratified_split(records, 42)
        config = settings.Config(shots=[1], policies=['none'], classifier_cs=[1.])
        fixture = (self.root / 'images', records, split, 'synthetic')
        with patch.object(p, 'check_device', return_value=torch.device('cpu')), \
                patch.object(p, 'prepare_data', return_value=fixture), \
                patch.object(p, 'load_backbone', return_value=MeanFeatures()):
            result = p.run(config, self.root, self.root / 'scratch', 'local')
            trial = next(path for path in (self.root / 'cache/experiments' / result['experiment']).glob('*.json')
                         if 'test' in json.loads(path.read_text()))
            old = json.loads(trial.read_text())
            old['test'] = {key: value for key, value in old['test'].items()
                           if key in {'accuracy', 'macro_f1', 'class_recall', 'confusion_matrix'}}
            trial.write_text(json.dumps(old))
            checkpoints = {path: path.stat().st_mtime_ns for path in (self.root / 'cache').rglob('*.npz')}
            with patch.object(p, 'load_backbone', side_effect=AssertionError('No model loading')), \
                    patch.object(p, 'fit_classifier', side_effect=AssertionError('No refitting')):
                p.run(config, self.root, self.root / 'scratch', 'local')
            self.assertIn('macro_precision', json.loads(trial.read_text())['test'])
            self.assertEqual(checkpoints, {path: path.stat().st_mtime_ns for path in checkpoints})

    def test_failed_atomic_write_preserves_previous_checkpoint(self):
        path = self.root / 'checkpoint.npz'
        path.write_bytes(b'previous')
        def fail(temporary):
            temporary.write_bytes(b'incomplete')
            raise RuntimeError('Interrupted')
        with self.assertRaises(RuntimeError):
            cache.atomic(path, fail)
        self.assertEqual(path.read_bytes(), b'previous')

    def test_portable_notebook_is_valid_and_helpers_match(self):
        from scripts.build_notebook import portable_source
        root = Path(__file__).resolve().parents[1]
        notebook = nbformat.read(root / 'EuroSAT_Representation_Benchmark.ipynb', as_version=4)
        nbformat.validate(notebook)
        cells = [c for c in notebook.cells if
                 'project_implementation' in c.metadata.get('tags', [])]
        self.assertEqual('\n\n'.join(c.source for c in cells), portable_source())
        namespace = {}
        for cell in cells:
            self.assertFalse(cell.metadata.get('jupyter', {}).get('source_hidden', False))
            exec(cell.source, namespace)
        self.assertTrue(callable(namespace['explore']))
        self.assertEqual(namespace['Config']().models, ['resnet18'])
        self.assertEqual({p.name for p in root.glob('*.ipynb')},
                         {'EuroSAT_Representation_Benchmark.ipynb', 'EuroSAT_Fine_Tuning.ipynb'})
        images = [o for c in notebook.cells for o in c.get('outputs', [])
                  if 'image/png' in o.get('data', {})]
        self.assertGreaterEqual(len(images), 6)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                compile(cell.source, 'notebook', 'exec')

    def test_notebook_drift_check_does_not_overwrite_edits(self):
        from scripts import build_notebook
        path = self.root / 'notebook.json'
        notebook = nbformat.read(build_notebook.NOTEBOOK, as_version=4)
        notebook.cells[2].source += '\n# A teammate edit that must survive a failed check.\n'
        nbformat.write(notebook, path)
        before = path.read_bytes()
        with patch.object(build_notebook, 'NOTEBOOK', path), \
                patch('sys.argv', ['build_notebook.py', '--check']):
            with self.assertRaisesRegex(SystemExit, 'stale'):
                build_notebook.main()
        self.assertEqual(path.read_bytes(), before)

    def test_fresh_colab_preflight_detects_gpu_and_rejects_missing_gpu(self):
        from subprocess import CompletedProcess
        root = Path(__file__).resolve().parents[1]
        notebook = nbformat.read(root / 'EuroSAT_Representation_Benchmark.ipynb', as_version=4)
        source = next(c.source for c in notebook.cells if c.cell_type == 'code')
        # Execute with empty globals: earlier imports cannot hide missing imports.
        with patch.dict('os.environ', {'EUROSAT_TARGET': 'colab_gpu', 'EUROSAT_MODE': 'smoke'}), \
                patch('pathlib.Path.is_dir', return_value=True), \
                patch('shutil.which', return_value='/usr/bin/nvidia-smi'), \
                patch('subprocess.run', return_value=CompletedProcess([], 0, 'Tesla T4\n', '')) as command:
            namespace = {}
            exec(compile(source, 'runtime-cell', 'exec'), namespace)
            self.assertEqual(namespace['TARGET'], 'colab_gpu')
            command.assert_called_once_with(
                ['/usr/bin/nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True)
        with patch.dict('os.environ', {'EUROSAT_TARGET': 'colab_gpu', 'EUROSAT_MODE': 'smoke'}), \
                patch('pathlib.Path.is_dir', return_value=True), \
                patch('shutil.which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'no NVIDIA GPU'):
                exec(compile(source, 'runtime-cell', 'exec'), {})

    def test_exploration_reports_measured_counts_and_rejects_leakage(self):
        from eurosat.exploration import dataset_profile, dataset_figures, FIGURE_NAMES
        records = self.images()
        split = data.stratified_split(records, 42)
        profile = dataset_profile(self.root / 'images', records, split)
        self.assertEqual(profile['images'], 200)
        self.assertEqual(profile['native_sizes'], {'24x24': 200})
        self.assertEqual(profile['partition_counts'], {'support': 120, 'dev': 40, 'test': 40})
        figures = dataset_figures(self.root / 'images', records, split, profile, self.root / 'figures')
        self.assertEqual(set(figures), set(FIGURE_NAMES))
        for exports in figures.values():
            self.assertTrue(Path(exports['png']).read_bytes().startswith(b'\x89PNG'))
            self.assertTrue(Path(exports['pdf']).read_bytes().startswith(b'%PDF'))
        sources = json.loads((self.root / 'figures/figure_sources.json').read_text())
        support_paths = {records[i]['path'] for i in split['support']}
        self.assertTrue(set(sources['class_examples']) <= support_paths)
        self.assertIn(sources['example'], support_paths)
        broken = {k: v.copy() for k, v in split.items()}
        broken['test'].append(split['support'][0])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            dataset_profile(self.root / 'images', records, broken)

    def test_merge_requires_compatible_complete_unique_trials(self):
        from eurosat.reporting import merge_results
        reports = []
        for model in ['resnet18', 'convnext_tiny']:
            config = settings.Config(models=[model], shots=[1], policies=['none'])
            manifest = {'config': asdict(config), 'dataset': 'same', 'trials': [{'1': [0]}],
                        'versions': {'torch': 'same'}, 'checkpoints': {model: settings.MODEL_IDS[model]}}
            rows = [{'model': model, 'policy': 'none', 'shots': 1, 'trial': 0,
                     'test': {'accuracy': .5, 'macro_f1': .4}, 'dev_accuracy': .6}]
            path = self.root / (model + '.json')
            path.write_text(json.dumps({'manifest': manifest, 'trials': rows}))
            reports.append(path)
        merged = merge_results(reports, self.root / 'combined')
        self.assertEqual(merged['models'], ['resnet18', 'convnext_tiny'])
        with self.assertRaisesRegex(ValueError, 'multiple exports'):
            merge_results([reports[0], reports[0]], self.root / 'bad')
        changed = json.loads(reports[1].read_text())
        changed['manifest']['config']['seed'] = 9
        reports[1].write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'different datasets'):
            merge_results(reports, self.root / 'bad')
        changed['manifest']['config']['seed'] = 42
        changed['trials'] = []
        reports[1].write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            merge_results(reports, self.root / 'bad')


if __name__ == '__main__':
    unittest.main()
