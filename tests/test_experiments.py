"""Proposal suite selection, paired budgets and resumable development tuning."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from eurosat import experiments
from eurosat.cache import key, prepare_storage
from eurosat.config import CLASSES, Config, MODEL_IDS
from eurosat.data import make_trials, stratified_split
from eurosat.evaluation import summarize
from eurosat.reporting import compare_budgets


class MeanFeatures(torch.nn.Module):
    def forward(self, images):
        return images.mean((2, 3))


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        old_hub = torch.hub.get_dir()
        self.addCleanup(torch.hub.set_dir, old_hub)
        prepare_storage(self.root)

    def test_full_part_d_plan_matches_required_grid_and_optional_cutout(self):
        config = experiments.part_d_config()
        self.assertEqual(config.models, ['dinov2_vits14'])
        self.assertEqual((config.trials, config.views, config.shots), (20, 4, [1, 5, 10]))
        self.assertIsNone(config.per_class_limit)
        self.assertEqual(len(config.policies), 10)
        optional = experiments.part_d_config(include_cutout=True)
        self.assertIn('cutout', optional.policies)
        optional.validate()
        for strengths in [{'noise': float('nan')}, {'noise': 0}, {'none': 1}]:
            with self.assertRaises(ValueError):
                replace(config, policy_strengths=strengths).validate()

    def test_resnet_baseline_reuses_locked_dino_configuration(self):
        configurations = []
        def recorded(config, *args):
            configurations.append(config)
            return {'output': str(self.root / (config.models[0] + str(config.views))),
                    'summary': [{'shots': 1, 'policy': 'flip', 'dev_accuracy': .7}]}
        with patch.object(experiments, 'prepare_data', return_value=(None, [], {}, 'same-data')), \
                patch.object(experiments, 'tune_strengths', return_value={'selected_strengths': {'rotation': .5}}), \
                patch.object(experiments, 'run', side_effect=recorded), \
                patch.object(experiments, 'create_report', return_value={'figures': {}, 'output': 'synthetic'}), \
                patch.object(experiments, 'merge_results', return_value={'output': str(self.root / 'combined')}), \
                patch.object(experiments, 'compare_budgets', return_value={}):
            experiments.run_part_d(self.root, self.root, 'local', mode='smoke')
        self.assertEqual([c.models for c in configurations], [['dinov2_vits14'], ['resnet18'], ['dinov2_vits14']])
        dino, resnet = asdict(configurations[0]), asdict(configurations[1])
        dino.pop('models'); resnet.pop('models')
        self.assertEqual(dino, resnet)
        self.assertEqual(dino['policy_strengths'], {'rotation': .5})

    def test_part_c_reuses_locked_strengths_without_selecting_them(self):
        configurations = []
        def recorded(config, *args):
            configurations.append(config)
            return {'output': str(self.root / (config.models[0] + str(config.views))),
                    'summary': [{'shots': 1, 'policy': 'flip', 'dev_accuracy': .7}]}
        with patch.object(experiments, 'prepare_data', return_value=(None, [], {}, 'same-data')), \
                patch.object(experiments, 'tune_strengths') as tune, \
                patch.object(experiments, 'run', side_effect=recorded), \
                patch.object(experiments, 'create_report', return_value={'figures': {}, 'output': 'synthetic'}), \
                patch.object(experiments, 'merge_results', return_value={'output': str(self.root / 'combined')}), \
                patch.object(experiments, 'compare_budgets', return_value={}):
            result = experiments.run_part_c(self.root, self.root, 'local', mode='smoke')
        tune.assert_not_called()
        self.assertEqual([c.models for c in configurations], [['vit_b_16'], ['resnet18'], ['vit_b_16']])
        vit, resnet = asdict(configurations[0]), asdict(configurations[1])
        vit.pop('models'); resnet.pop('models')
        self.assertEqual(vit, resnet)
        self.assertEqual(vit['policy_strengths'], experiments.LOCKED_STRENGTHS)
        self.assertEqual((configurations[2].shots, configurations[2].views), ([1], 9))
        self.assertEqual(Path(result['output']).parent.name, 'part-c')

    def test_part_c_and_part_d_stay_separate(self):
        vit, dino = experiments.part_c_config('smoke'), experiments.part_d_config('smoke')
        with patch.object(experiments, 'prepare_data', side_effect=AssertionError('checked before data')):
            with self.assertRaisesRegex(ValueError, 'Part D primary grid'):
                experiments.run_part_d(self.root, self.root, 'local', config=vit)
            with self.assertRaisesRegex(ValueError, 'Part C primary grid'):
                experiments.run_part_c(self.root, self.root, 'local', config=dino)
            with self.assertRaisesRegex(ValueError, 'strengths locked by Part D'):
                experiments.run_part_c(self.root, self.root, 'local', config=replace(vit, policy_strengths={}))

    def test_part_c_strength_review_keeps_the_locked_values(self):
        locked = dict(experiments.LOCKED_STRENGTHS)
        selected = {'brightness_contrast': 1., 'noise': 1.5, 'blur': .5, 'erasing': 1.}
        with patch.object(experiments, 'tune_strengths',
                          return_value={'selected_strengths': selected, 'scores': [], 'query_labels_used': 0,
                                        'output': str(self.root / 'development')}) as tune:
            result = experiments.review_part_c_strengths(self.root, self.root, 'local', mode='smoke')
        self.assertEqual(tune.call_args.args[0].models, ['vit_b_16'])
        self.assertEqual(result['review']['noise'], {'vit_selected': 1.5, 'locked': .5})
        self.assertEqual(experiments.LOCKED_STRENGTHS, locked)
        self.assertTrue((self.root / 'development/part-c-strength-review.json').exists())

    def test_vit_adapter_returns_frozen_768_dimensional_class_tokens(self):
        from torchvision import models as tv
        from eurosat.models import load_backbone
        with patch.object(tv, 'vit_b_16', lambda weights: tv.VisionTransformer(
                image_size=224, patch_size=16, num_layers=1, num_heads=12, hidden_dim=768, mlp_dim=3072)):
            model = load_backbone('vit_b_16', torch.device('cpu'))
        self.assertIsInstance(model.heads, torch.nn.Identity)
        self.assertFalse(model.training)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        images = torch.zeros(2, 3, 224, 224)
        with torch.inference_mode():
            output = model(images)
            tokens = model.encoder(torch.cat([model.class_token.expand(2, -1, -1), model._process_input(images)], dim=1))
        self.assertEqual(tuple(output.shape), (2, 768))
        self.assertEqual(tuple(tokens.shape), (2, 197, 768))
        self.assertTrue(torch.equal(output, tokens[:, 0]))

    def test_nine_view_method_selection_ignores_test_scores(self):
        primary = {'summary': [
            {'shots': 1, 'policy': 'flip', 'dev_accuracy': .8, 'accuracy': .1},
            {'shots': 1, 'policy': 'noise', 'dev_accuracy': .7, 'accuracy': 1.},
            {'shots': 1, 'policy': 'none', 'dev_accuracy': 1., 'accuracy': 1.},
            {'shots': 5, 'policy': 'rotation', 'dev_accuracy': 1., 'accuracy': 1.}]}
        self.assertEqual(experiments.select_nine_view_methods(primary, 1), ['flip'])

    def test_strength_selection_is_development_only_and_resumes_without_model(self):
        records = []
        for label, name in enumerate(CLASSES):
            for i in range(20):
                relative = f'{name}/{i}.png'
                path = self.root / 'images' / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (16, 16), (label * 23, i * 7, (label + i) * 3)).save(path)
                records.append({'path': relative, 'label': label, 'pixels_sha256': key([label, i])})
        split = stratified_split(records, 42)
        config = Config(models=['dinov2_vits14'], policies=['none', 'noise'], shots=[1],
                        trials=2, views=1, classifier_cs=[1.])
        fixture = self.root / 'images', records, split, 'synthetic'
        with patch.object(experiments, 'prepare_data', return_value=fixture), \
                patch.object(experiments, 'check_device', return_value=torch.device('cpu')), \
                patch.object(experiments, 'load_backbone', return_value=MeanFeatures()):
            selected = experiments.tune_strengths(config, self.root, self.root, 'local', candidates=(.5, 1.))
            self.assertEqual(selected['query_labels_used'], 0)
            self.assertEqual(selected['development_labels'], len(split['dev']))
            self.assertEqual(len(selected['scores']), 4)
            self.assertTrue(all('test' not in row for row in selected['scores']))
            with patch.object(experiments, 'load_backbone', side_effect=AssertionError('cached selection')):
                resumed = experiments.tune_strengths(config, self.root, self.root, 'local', candidates=(.5, 1.))
            self.assertEqual(selected, resumed)

    def test_strength_changes_views_without_changing_original_support_or_seeds(self):
        from eurosat.augment import image_tensor
        path = self.root / 'image.png'
        Image.new('RGB', (32, 32), (80, 100, 120)).save(path)
        weak = image_tensor(path, 'noise', 42, .5)
        strong = image_tensor(path, 'noise', 42, 1.5)
        self.assertFalse(torch.equal(weak, strong))
        self.assertTrue(torch.equal(strong, image_tensor(path, 'noise', 42, 1.5)))
        self.assertTrue(torch.equal(image_tensor(path, 'none', 42), image_tensor(path, 'none', 42, 1.5)))

    def budget_fixture(self, views, shots):
        config = Config(models=['dinov2_vits14'], policies=['none', 'repeat', 'flip'],
                        shots=shots, trials=2, views=views)
        records = [{'label': label} for label in range(10) for _ in range(20)]
        split = stratified_split(records, config.seed)
        trials = make_trials(records, split, config)
        rows = []
        for policy in config.policies:
            for shot in shots:
                for trial in range(config.trials):
                    accuracy = (.4 + .02 * shot) if policy == 'none' else .5 + .01 * views
                    rows.append({'model': 'dinov2_vits14', 'policy': policy, 'shots': shot, 'trial': trial,
                                 'test': {'accuracy': accuracy, 'macro_f1': accuracy}, 'dev_accuracy': .6,
                                 'original_support_images': 10 * shot,
                                 'classifier_rows': 10 * shot * (1 if policy == 'none' else views + 1)})
        return {'manifest': {'config': asdict(config), 'dataset': 'same', 'versions': {'fixture': True},
                             'checkpoints': {'dinov2_vits14': MODEL_IDS['dinov2_vits14']}, 'trials': trials},
                'trials': rows, 'summary': summarize(rows)}

    def test_budget_arithmetic_and_identical_original_support_requirement(self):
        primary, nine = self.budget_fixture(4, [1, 5, 10]), self.budget_fixture(9, [1])
        a, b = self.root / 'four.json', self.root / 'nine.json'
        a.write_text(json.dumps(primary)); b.write_text(json.dumps(nine))
        result = compare_budgets(a, b, self.root / 'comparisons')
        import pandas as pd
        rows = pd.read_csv(Path(result['output']) / 'budget-comparisons.csv')
        effect = rows[rows.comparison == 'nine_vs_four_views']
        np.testing.assert_allclose(effect.difference_percentage_points, 5.)
        matched = rows[rows.comparison != 'nine_vs_four_views']
        self.assertTrue((matched.classifier_rows == matched.reference_classifier_rows).all())
        self.assertTrue((matched.original_images < matched.reference_original_images).all())
        nine['manifest']['trials'][0]['1'][0] += 1
        b.write_text(json.dumps(nine))
        with self.assertRaisesRegex(ValueError, 'support images must be identical'):
            compare_budgets(a, b, self.root / 'bad')
