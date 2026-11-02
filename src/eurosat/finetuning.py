"""Optional partial-adaptation extension, isolated from frozen-feature results."""
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from .cache import atomic, key, prepare_storage, write_json
from .config import CLASSES, Config, MODEL_IDS
from .data import make_trials, prepare_data
from .evaluation import score
from .features import Images
from .models import load_backbone
from .runtime import check_device, environment

FINETUNE_VERSION = '1'
FEATURE_DIMENSIONS = {'resnet18': 512, 'convnext_tiny': 768, 'vit_b_16': 768, 'dinov2_vits14': 384}


@dataclass
class FineTuneConfig:
    model: str = 'dinov2_vits14'
    mode: str = 'smoke'
    seed: int = 42
    shots: list = field(default_factory=lambda: [1, 5, 10])
    trials: int = 1
    epochs: int = 2
    batch_size: int = 16
    patience: int = 3
    head_lr: float = .001
    backbone_lr: float = .00001
    weight_decay: float = .0001
    views: int = 1
    policy: str = 'flip'
    strategy: str = 'last_block'
    include_frozen_control: bool = True
    last_blocks: int = 1

    @classmethod
    def full(cls, model='dinov2_vits14'):
        return cls(model=model, mode='full', trials=20, epochs=10, views=4)

    def validate(self):
        if self.model not in MODEL_IDS or self.mode not in {'smoke', 'full'}:
            raise ValueError('Choose a supported model and smoke/full mode')
        if self.strategy != 'last_block':
            raise ValueError('Only partial last-block fine-tuning is enabled')
        if self.last_blocks not in {1, 2} or (self.model != 'dinov2_vits14' and self.last_blocks != 1):
            raise ValueError('Choose one final stage, or one/two blocks for DINOv2')
        if not self.shots or min(self.shots) < 1 or len(set(self.shots)) != len(self.shots):
            raise ValueError('Choose unique positive original-shot budgets')
        if min(self.trials, self.epochs, self.batch_size, self.patience) < 1 or self.views < 0:
            raise ValueError('Trials, epochs, batch size and patience must be positive; views nonnegative')
        if not all(np.isfinite(v) for v in [self.head_lr, self.backbone_lr, self.weight_decay]):
            raise ValueError('Optimizer settings must be finite')
        if min(self.head_lr, self.backbone_lr) <= 0 or self.weight_decay < 0:
            raise ValueError('Learning rates must be positive and weight decay nonnegative')
        if self.policy not in {'none', 'repeat', 'flip', 'rotation', 'translation', 'scale',
                               'brightness_contrast', 'noise', 'blur', 'erasing', 'cutout'}:
            raise ValueError('Unsupported training augmentation')

    def data_config(self):
        return Config(mode=self.mode, seed=self.seed, models=[self.model], shots=self.shots,
                      trials=self.trials, per_class_limit=40 if self.mode == 'smoke' else None)


class AdaptedClassifier(nn.Module):
    def __init__(self, backbone, dimension):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(dimension, len(CLASSES))

    def forward(self, images):
        # Same representation normalization in the matched control and fine-tuned model.
        return self.head(nn.functional.normalize(self.backbone(images).flatten(1), dim=1))


def build_adapted_model(config, strategy, device, seed):
    """Use identical pretrained weights and head initialization across conditions."""
    config.validate()
    if strategy not in {'last_block', 'frozen_head'}:
        raise ValueError('Choose partial last_block adaptation or the frozen_head control')
    torch.manual_seed(seed)
    backbone = load_backbone(config.model, device)
    torch.manual_seed(seed)
    model = AdaptedClassifier(backbone, FEATURE_DIMENSIONS[config.model]).to(device)
    if strategy == 'last_block':
        if config.model == 'resnet18':
            blocks = [backbone.layer4]
        elif config.model == 'convnext_tiny':
            blocks = [backbone.features[-1], backbone.classifier[0]]
        elif config.model == 'vit_b_16':
            blocks = [backbone.encoder.layers[-1], backbone.encoder.ln]
        else:
            blocks = [*backbone.blocks[-config.last_blocks:], backbone.norm]
        for block in blocks:
            block.requires_grad_(True)
    return model


def optimizer_for(model, config):
    groups = [{'params': model.head.parameters(), 'lr': config.head_lr}]
    parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    if parameters:
        groups.append({'params': parameters, 'lr': config.backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def trainable_state(model):
    """Frozen tensors reload from the pinned pretrained checkpoint; save changed tensors."""
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}


def restore_trainable(model, state):
    expected = {name for name, value in model.named_parameters() if value.requires_grad}
    if set(state) != expected:
        raise ValueError('Checkpoint parameters do not match this training condition')
    model.load_state_dict(state, strict=False)


def evaluate_classifier(model, loader, device):
    model.eval()
    labels, predictions, loss_sum, count = [], [], 0., 0
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            loss_sum += nn.functional.cross_entropy(logits, targets.to(device), reduction='sum').item()
            labels.extend(targets.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
            count += len(targets)
    return {'loss': loss_sum / count, 'accuracy': float(np.mean(np.array(labels) == predictions)),
            'labels': labels, 'predictions': predictions}


def fit_support(model, config, training, development, device, checkpoint):
    """Resume at epoch boundaries; only development data choose the retained epoch."""
    optimizer = optimizer_for(model, config)
    history, best, best_state, stale, start = [], None, None, 0, 0
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        restore_trainable(model, saved['current'])
        optimizer.load_state_dict(saved['optimizer'])
        history, best, best_state = saved['history'], saved['best'], saved['best_state']
        start, stale = saved['next_epoch'], saved['stale']
        print(f'Resuming at epoch {start + 1}', flush=True)
    for epoch in range(start, config.epochs):
        if stale >= config.patience:
            break
        generator = torch.Generator().manual_seed(config.seed + epoch)
        loader = DataLoader(training, batch_size=config.batch_size, shuffle=True,
                            generator=generator, num_workers=0)
        model.train()
        # Keep pretrained BatchNorm statistics/dropout fixed for tiny support sets.
        # Evaluation mode does not disable gradients through the unfrozen block.
        model.backbone.eval()
        loss_sum, correct, count = 0., 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += int((logits.argmax(1) == labels).sum().item())
            count += len(labels)
        dev = evaluate_classifier(model, development, device)
        row = {'epoch': epoch + 1, 'train_loss': loss_sum / count, 'train_accuracy': correct / count,
               'dev_loss': dev['loss'], 'dev_accuracy': dev['accuracy']}
        history.append(row)
        if best is None or (row['dev_accuracy'], -row['dev_loss']) > (best['dev_accuracy'], -best['dev_loss']):
            best, best_state, stale = row, trainable_state(model), 0
        else:
            stale += 1
        saved = {'current': trainable_state(model), 'optimizer': optimizer.state_dict(),
                 'history': history, 'best': best, 'best_state': best_state,
                 'next_epoch': epoch + 1, 'stale': stale}
        atomic(checkpoint, lambda p: torch.save(saved, p))
        print(f"Epoch {epoch + 1}: train={row['train_accuracy']:.3f}, dev={row['dev_accuracy']:.3f}", flush=True)
    restore_trainable(model, best_state)
    return history, best


def run_finetuning(config, storage, scratch, target='colab_gpu'):
    config.validate()
    device = check_device(target)
    storage = prepare_storage(storage)
    root, records, split, dataset_id = prepare_data(storage, scratch, config.data_config())
    trials = make_trials(records, split, config.data_config())
    manifest = {'study': 'optional-finetuning', 'version': FINETUNE_VERSION, 'config': asdict(config),
                'dataset': dataset_id, 'trials': trials, 'checkpoint': MODEL_IDS[config.model],
                'versions': environment(), 'normalization': 'L2 embeddings',
                'pretrained_batchnorm_and_dropout': 'evaluation mode in all training conditions'}
    identifier = key(manifest)
    directory = storage / 'cache/finetuning' / identifier
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'manifest.json', manifest)
    loaders = {part: DataLoader(Images(root, records, [(i, 0) for i in split[part]], 'none', config.seed),
                                batch_size=config.batch_size, num_workers=0)
               for part in ['dev', 'test']}
    strategies = ['frozen_head', config.strategy] if config.include_frozen_control else [config.strategy]
    rows = []
    for trial_number, trial in enumerate(trials):
        for shots in sorted(config.shots):
            original = trial[str(shots)]
            datasets = [Images(root, records, [(i, 0) for i in original], 'none', config.seed)]
            if config.views and config.policy != 'none':
                datasets.append(Images(root, records, [(i, v) for i in original for v in range(config.views)],
                                       'none' if config.policy == 'repeat' else config.policy, config.seed))
            training = ConcatDataset(datasets)
            for strategy in strategies:
                name = f'{strategy}-{shots}shot-trial{trial_number}'
                result_path = directory / f'{name}.json'
                if result_path.exists():
                    rows.append(json.loads(result_path.read_text()))
                    print('Completed fine-tuning result reused:', name, flush=True)
                    continue
                paired = replace(config, seed=config.seed + trial_number)
                started = time.perf_counter()
                model = build_adapted_model(paired, strategy, device, paired.seed)
                history, best = fit_support(model, paired, training, loaders['dev'], device, directory / f'{name}.pt')
                # The query partition is evaluated once, after development selection.
                heldout = evaluate_classifier(model, loaders['test'], device)
                result = {'model': config.model, 'strategy': strategy, 'shots': shots, 'trial': trial_number,
                          'original_support_images': len(original), 'training_rows': len(training),
                          'selected_epoch': best['epoch'], 'dev_accuracy': best['dev_accuracy'],
                          'test': score(heldout['labels'], heldout['predictions']), 'history': history,
                          'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                          'total_parameters': sum(p.numel() for p in model.parameters()),
                          'seconds': time.perf_counter() - started, 'device': str(device)}
                write_json(result_path, result)
                rows.append(result)
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'mps':
                    torch.mps.empty_cache()
    output = storage / 'outputs/finetuning' / identifier
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'results.json', {'manifest': manifest, 'trials': rows})
    print('Optional fine-tuning results:', output, flush=True)
    return {'output': str(output), 'manifest': manifest, 'trials': rows}


def run_finetuning_study(config, storage, scratch, target='colab_gpu'):
    """Prespecified adaptation × augmentation study, with a matched repeat control."""
    config.validate()
    if config.model != 'dinov2_vits14' or not config.include_frozen_control or config.views < 1:
        raise ValueError('The DINOv2 study requires paired controls and at least one added view')
    conditions = [('none', 0), ('repeat', config.views), ('flip', config.views)]
    runs, rows = [], []
    for policy, views in conditions:
        result = run_finetuning(replace(config, policy=policy, views=views), storage, scratch, target)
        runs.append(result['manifest'])
        rows.extend({**row, 'augmentation': policy, 'added_views': views,
                     'last_blocks': config.last_blocks} for row in result['trials'])
    manifest = {'study': 'dinov2-partial-adaptation-factorial', 'config': asdict(config), 'runs': runs}
    output = Path(storage) / 'outputs/finetuning' / key(manifest)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'results.json', {'manifest': manifest, 'trials': rows})
    return {'output': str(output), 'manifest': manifest, 'trials': rows}
