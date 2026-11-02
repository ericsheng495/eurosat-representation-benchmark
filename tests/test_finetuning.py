"""Fine-tuning parameter scopes, epoch recovery and held-out isolation."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eurosat import finetuning as ft
from eurosat.finetuning_reporting import finetuning_contrasts, finetuning_tables


class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.early = nn.Linear(4, 4)
        self.layer4 = nn.Linear(4, 512)

    def forward(self, x):
        return self.layer4(self.early(x))


class FineTuningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        torch.manual_seed(3)
        self.backbone = SmallResNet().requires_grad_(False)
        self.config = ft.FineTuneConfig(model='resnet18', shots=[1], epochs=3, batch_size=8)
        generator = torch.Generator().manual_seed(2)
        self.data = TensorDataset(torch.randn(20, 4, generator=generator), torch.arange(20) % 10)
        self.dev = DataLoader(self.data, batch_size=8)

    def model(self, strategy):
        with patch.object(ft, 'load_backbone', side_effect=lambda *args: deepcopy(self.backbone)):
            return ft.build_adapted_model(self.config, strategy, torch.device('cpu'), 42)

    def test_control_and_adaptation_have_identical_heads_and_correct_parameter_scopes(self):
        control, adapted = self.model('frozen_head'), self.model('last_block')
        self.assertTrue(torch.equal(control.head.weight, adapted.head.weight))
        self.assertFalse(any(p.requires_grad for p in control.backbone.parameters()))
        self.assertFalse(any(p.requires_grad for p in adapted.backbone.early.parameters()))
        self.assertTrue(all(p.requires_grad for p in adapted.backbone.layer4.parameters()))
        before = deepcopy(adapted.state_dict())
        ft.fit_support(adapted, self.config, self.data, self.dev, torch.device('cpu'), self.root / 'trained.pt')
        self.assertTrue(torch.equal(before['backbone.early.weight'], adapted.backbone.early.weight))
        self.assertFalse(torch.equal(before['backbone.layer4.weight'], adapted.backbone.layer4.weight))

    def test_full_backbone_training_is_disabled(self):
        with self.assertRaisesRegex(ValueError, 'Only partial'):
            ft.FineTuneConfig(strategy='full').validate()
        with self.assertRaisesRegex(ValueError, 'partial'):
            self.model('full')

    def test_dino_defaults_and_two_block_scope(self):
        self.assertEqual(ft.FineTuneConfig().model, 'dinov2_vits14')
        self.assertEqual(ft.FineTuneConfig.full().last_blocks, 1)
        backbone = nn.Module()
        backbone.blocks = nn.ModuleList([nn.Linear(384, 384) for _ in range(3)])
        backbone.norm = nn.LayerNorm(384)
        backbone.requires_grad_(False)
        with patch.object(ft, 'load_backbone', return_value=backbone):
            adapted = ft.build_adapted_model(ft.FineTuneConfig(last_blocks=2), 'last_block', torch.device('cpu'), 42)
        self.assertFalse(any(p.requires_grad for p in adapted.backbone.blocks[0].parameters()))
        self.assertTrue(all(p.requires_grad for block in adapted.backbone.blocks[1:] for p in block.parameters()))
        for config in [ft.FineTuneConfig(last_blocks=3), ft.FineTuneConfig(model='resnet18', last_blocks=2)]:
            with self.assertRaises(ValueError):
                config.validate()

    def test_factorial_study_and_contrasts_are_paired(self):
        calls = []
        def run(config, storage, scratch, target):
            calls.append(config)
            rows = [{'strategy': strategy, 'shots': shots, 'trial': 0,
                     'test': {'accuracy': .4 + .1 * (strategy == 'last_block') + .01 * shots
                              + .02 * (config.policy == 'flip'), 'macro_f1': .4}}
                    for strategy in ['frozen_head', 'last_block'] for shots in config.shots]
            return {'manifest': {'config': ft.asdict(config)}, 'trials': rows}
        config = ft.FineTuneConfig(views=4)
        with patch.object(ft, 'run_finetuning', side_effect=run):
            result = ft.run_finetuning_study(config, self.root, self.root, 'local')
        self.assertEqual([(c.policy, c.views) for c in calls], [('none', 0), ('repeat', 4), ('flip', 4)])
        self.assertEqual(len(result['trials']), 18)
        contrasts = finetuning_contrasts(result['trials'])
        adaptation = [r for r in contrasts if r['comparison'] == 'adaptation over frozen head']
        self.assertEqual(len(adaptation), 9)
        self.assertTrue(all(abs(r['accuracy_gain_pp'] - 10) < 1e-10 for r in adaptation))
        budgets = [r for r in contrasts if r['comparison'] == '1 original + 4 flips over 5 originals']
        self.assertEqual(len(budgets), 2)
        self.assertTrue(all(abs(r['accuracy_gain_pp'] + 2) < 1e-10 for r in budgets))

    def test_other_backbone_scopes_unfreeze_only_the_final_stage_and_norm(self):
        class Backbone(nn.Module):
            pass
        for name in ['convnext_tiny', 'vit_b_16', 'dinov2_vits14']:
            backbone = Backbone()
            if name == 'convnext_tiny':
                backbone.features = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 768))
                backbone.classifier = nn.Sequential(nn.LayerNorm(768), nn.Flatten(), nn.Identity())
                expected = {'features.1.weight', 'features.1.bias', 'classifier.0.weight', 'classifier.0.bias'}
            elif name == 'vit_b_16':
                backbone.encoder = Backbone()
                backbone.encoder.layers = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 768))
                backbone.encoder.ln = nn.LayerNorm(768)
                expected = {'encoder.layers.1.weight', 'encoder.layers.1.bias', 'encoder.ln.weight', 'encoder.ln.bias'}
            else:
                backbone.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 384)])
                backbone.norm = nn.LayerNorm(384)
                expected = {'blocks.1.weight', 'blocks.1.bias', 'norm.weight', 'norm.bias'}
            backbone.requires_grad_(False)
            config = ft.FineTuneConfig(model=name)
            with patch.object(ft, 'load_backbone', return_value=backbone):
                adapted = ft.build_adapted_model(config, 'last_block', torch.device('cpu'), 42)
            self.assertEqual({n for n, p in adapted.backbone.named_parameters() if p.requires_grad}, expected)
            self.assertEqual(adapted.head.in_features, ft.FEATURE_DIMENSIONS[name])

    def test_interrupted_epoch_resume_reproduces_uninterrupted_training(self):
        reference = self.model('last_block')
        history, selected = ft.fit_support(reference, self.config, self.data, self.dev,
                                           torch.device('cpu'), self.root / 'reference.pt')
        interrupted = self.model('last_block')
        actual_atomic = ft.atomic
        calls = []
        def interrupt(path, write):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('Disconnected')
            return actual_atomic(path, write)
        with patch.object(ft, 'atomic', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'Disconnected'):
                ft.fit_support(interrupted, self.config, self.data, self.dev,
                               torch.device('cpu'), self.root / 'resume.pt')
        checkpoint = torch.load(self.root / 'resume.pt', weights_only=True)
        self.assertEqual(checkpoint['next_epoch'], 1)
        resumed = self.model('last_block')
        resumed_history, resumed_selected = ft.fit_support(resumed, self.config, self.data, self.dev,
                                                          torch.device('cpu'), self.root / 'resume.pt')
        self.assertEqual(history, resumed_history)
        self.assertEqual(selected, resumed_selected)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)

    def test_query_is_scored_only_after_epoch_selection_and_complete_jobs_skip_models(self):
        import json
        from PIL import Image
        from eurosat.cache import key
        from eurosat.data import stratified_split
        records = []
        for label in range(10):
            for number in range(20):
                name = f'{label}/{number}.png'
                path = self.root / 'images' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (16, 16), (label * 23, number * 7, label + number)).save(path)
                records.append({'label': label, 'path': name, 'pixels_sha256': key([label, number])})
        split = stratified_split(records, 42)
        class PixelBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 3)
            def forward(self, images):
                return self.linear(images.mean((2, 3)))
        def model(config, strategy, device, seed):
            torch.manual_seed(seed)
            backbone = PixelBackbone().requires_grad_(strategy != 'frozen_head')
            return ft.AdaptedClassifier(backbone, 3)
        config = ft.FineTuneConfig(shots=[1], epochs=2, views=1, policy='repeat')
        original_hub = torch.hub.get_dir()
        self.addCleanup(torch.hub.set_dir, original_hub)
        actual_score = ft.score
        with patch.object(ft, 'prepare_data', return_value=(self.root / 'images', records, split, 'synthetic')), \
                patch.object(ft, 'check_device', return_value=torch.device('cpu')), \
                patch.object(ft, 'build_adapted_model', side_effect=model), \
                patch.object(ft, 'score', wraps=actual_score) as scorer:
            result = ft.run_finetuning(config, self.root, self.root / 'scratch', 'local')
            self.assertEqual(scorer.call_count, 2)  # One held-out score per completed condition.
            self.assertTrue(all(r['training_rows'] == 20 and r['original_support_images'] == 10
                                for r in result['trials']))
            self.assertTrue(all('test' not in row for r in result['trials'] for row in r['history']))
            with patch.object(ft, 'build_adapted_model', side_effect=AssertionError('Completed jobs skip model loading')):
                rerun = ft.run_finetuning(config, self.root, self.root / 'scratch', 'local')
            self.assertEqual(result['trials'], rerun['trials'])
            self.assertEqual(scorer.call_count, 2)

    def test_finetuning_notebook_embeds_readable_code_and_compiles(self):
        from scripts.build_finetuning_notebook import NOTEBOOK, implementation_blocks
        notebook = nbformat.read(NOTEBOOK, 4)
        nbformat.validate(notebook)
        cells = [c for c in notebook.cells if 'finetuning_implementation' in c.metadata.get('tags', [])]
        self.assertEqual([c.source for c in cells], [source for _, source in implementation_blocks()])
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                compile(cell.source, 'fine-tuning-cell', 'exec')

    def test_compact_notebook_has_unique_imports_and_only_required_helpers(self):
        import ast
        from collections import Counter
        from scripts.build_finetuning_notebook import build
        bindings, definitions = [], []
        for cell in build(write=False).cells:
            if cell.cell_type != 'code':
                continue
            for node in ast.parse(cell.source).body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    bindings.extend(a.asname or a.name.split('.')[0] for a in node.names)
                elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    definitions.append(node.name)
        self.assertEqual([name for name, n in Counter(bindings).items() if n > 1], [])
        self.assertEqual([name for name, n in Counter(definitions).items() if n > 1], [])
        self.assertTrue({'Images', 'score', 'metrics_from_confusion', 'save_figure'} <= set(definitions))
        self.assertFalse({'extract', 'dataset_figures', 'explore', 'fit_classifier'} & set(definitions))

    def test_shared_reporting_tables_preserve_scores_and_paired_effects(self):
        rows = []
        for strategy in ['frozen_head', 'last_block']:
            for trial in range(2):
                metrics = ft.score(list(range(10)), [0] * 10 if strategy == 'frozen_head' else list(range(10)))
                rows.append({'strategy': strategy, 'augmentation': 'none', 'shots': 1, 'trial': trial,
                             'selected_epoch': 2, 'trainable_parameters': 10, 'total_parameters': 100,
                             'seconds': 1., 'test': metrics})
        tables = finetuning_tables(rows)
        adapted = tables['summary'].query('strategy == "last_block"').iloc[0]
        self.assertEqual(adapted['accuracy_mean'], 1.)
        self.assertEqual(adapted['accuracy_std'], 0.)
        self.assertTrue((tables['contrasts']['accuracy_gain_pp'] == 90.).all())
        self.assertEqual(len(tables['class_recall']), 40)
