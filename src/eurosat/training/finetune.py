"""Fine-tuning for baseline 3: train each ImageNet-pretrained model further on the six base classes.

One recipe for every model (research_info.md, section 4.3):
  - start from a fresh copy of the ImageNet weights and add a new 6-class head that starts at zero
  - train the whole network on the base-class train part (10,200 images), same image order for every run
  - AdamW, weight decay 0.05 (none on biases, normalization weights, layer scale, position embeddings, class tokens)
  - 10 epochs, batch 64; the learning rate changes once per epoch: epoch 1 at 0.1x (warmup), then cosine
  - mixed precision on CUDA; no augmentation, no dropout or stochastic depth in any model
  - learning rate chosen from 10 candidates (1e-6 to 1e-3) by validation accuracy (ties keep the smaller one);
    the best epoch of the chosen run is kept
  - the kept model's 6-class test accuracy is reported as a check

Only base-class images are read; the four evaluated classes are never seen.
Each learning-rate run is saved when it finishes, so an interrupted job resumes where it stopped.
Saved runs are reused only if they were trained with this exact recipe; a changed recipe trains again.
Weights stay in shared storage, not in Git.
"""
import hashlib
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..cache import digest_file, key, prepare_storage, write_json
from ..config import BASE_CLASSES, FINETUNED_WEIGHTS_DIR, MODEL_IDS
from ..models import fine_tuning_network, finetuned_files
from ..preprocessing import Images, load_source, validate_manifest

RECIPE = {
    'start': 'ImageNet-1K weights + new 6-class head', 'head_init': 'zeros', 'trained': 'whole network',
    'optimizer': 'AdamW', 'weight_decay': 0.05,
    'no_weight_decay': 'biases, normalization weights, layer scale, position embeddings, class tokens',
    'learning_rates': [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
    'warmup_epochs': 1, 'warmup_start_factor': 0.1,
    'schedule': 'updated once per epoch: linear warmup from 0.1x, then cosine over the remaining epochs',
    'epochs': 10, 'batch_size': 64, 'mixed_precision': 'float16 on CUDA', 'augmentation': 'none',
    'random_regularizers': 'none (ConvNeXt stochastic depth off)',
    'preprocessing': 'RGB, bilinear resize to 224x224, ImageNet normalization', 'seed': 42,
    'selection': 'best validation accuracy over learning rates and epochs; ties keep the smaller learning rate',
}
NO_DECAY = ('layer_scale', 'pos_embedding', 'class_token', 'pos_embed', 'cls_token')


def recipe_id(recipe):
    """Identifies how one learning-rate run was trained (the list of candidate rates is not part of a run)."""
    return key({name: value for name, value in recipe.items() if name != 'learning_rates'})


def finetuned_weights_dir(storage, smoke=False):
    """Fine-tuned weights folder. Smoke runs use a separate folder and never touch the real weights."""
    folder = Path(storage) / FINETUNED_WEIGHTS_DIR
    return folder / 'smoke' if smoke else folder


def parameter_groups(network, weight_decay):
    decay, no_decay = [], []
    for name, parameter in network.named_parameters():
        skip = parameter.ndim <= 1 or name.rsplit('.', 1)[-1] in NO_DECAY
        (no_decay if skip else decay).append(parameter)
    return [{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]


def base_class_data(storage, scratch, manifest_path):
    """Image root and records (label 0-5 in BASE_CLASSES order) for the base train / val / test parts."""
    source = load_source(storage, scratch)
    saved = validate_manifest(manifest_path, source).payload
    by_path = {row['path']: row for row in source.records}
    parts = {}
    for part in ['train', 'val', 'test']:
        parts[part] = [dict(by_path[f'{cls}/{name}'], label=label)
                       for label, cls in enumerate(BASE_CLASSES) for name in saved['base'][part][cls]]
    return source.root, parts, digest_file(manifest_path)


def make_loader(root, records, batch_size, shuffle, seed, workers, device):
    dataset = Images(root, records, [(i, 0) for i in range(len(records))], 'none', seed)
    generator = torch.Generator().manual_seed(seed) if shuffle else None  # Same order for every run.
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=workers,
                      pin_memory=device.type == 'cuda', persistent_workers=workers > 0)


def evaluate(network, loader, device, use_amp, max_batches=None):
    network.eval()
    loss_sum, correct, seen = 0.0, 0, 0
    with torch.inference_mode():
        for batch, (images, labels) in enumerate(loader):
            if max_batches and batch >= max_batches:
                break
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = network(images)
            loss_sum += nn.functional.cross_entropy(logits.float(), labels, reduction='sum').item()
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.numel()
    return loss_sum / seen, correct / seen


def epoch_factor(epoch, recipe):
    """Learning-rate multiplier for a 0-based epoch: linear warmup from the start factor, then cosine."""
    warmup, epochs, start = recipe['warmup_epochs'], recipe['epochs'], recipe['warmup_start_factor']
    if epoch < warmup:
        return start + (1 - start) * epoch / warmup
    return .5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, epochs - warmup)))


def train_run(name, lr, loaders, device, recipe, max_batches=None):
    """One learning rate. Returns the best-validation state (on the CPU) and the run record."""
    use_amp = device.type == 'cuda'
    torch.manual_seed(recipe['seed'])  # Same random state for every learning rate.
    network = fine_tuning_network(name).to(device)
    optimizer = torch.optim.AdamW(parameter_groups(network, recipe['weight_decay']), lr=lr)
    steps = min(len(loaders['train']), max_batches or len(loaders['train']))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: epoch_factor(epoch, recipe))
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    history = {key: [] for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr']}
    best_state, best_acc, best_epoch, started = None, -1.0, 0, time.perf_counter()

    for epoch in range(1, recipe['epochs'] + 1):
        network.train()
        history['lr'].append(optimizer.param_groups[0]['lr'])
        loss_sum, correct, seen = 0.0, 0, 0
        for batch, (images, labels) in enumerate(loaders['train']):
            if batch >= steps:
                break
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = network(images)
                loss = nn.functional.cross_entropy(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * labels.numel()
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.numel()
        schedule.step()  # Once per epoch.
        val_loss, val_acc = evaluate(network, loaders['val'], device, use_amp, max_batches)
        for metric, value in [('train_loss', loss_sum / seen), ('train_acc', correct / seen),
                              ('val_loss', val_loss), ('val_acc', val_acc)]:
            history[metric].append(value)
        if val_acc > best_acc:  # Selection uses validation only, never test.
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().to('cpu', copy=True) for k, v in network.state_dict().items()}
    del network, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_state, {'learning_rate': lr, 'best_epoch': best_epoch, 'best_val_accuracy': best_acc,
                        'minutes': round((time.perf_counter() - started) / 60, 2), 'recipe_id': recipe_id(recipe),
                        'history': history}


def fine_tune_model(name, data, weights_dir, device, recipe, workers=4, max_batches=None, overwrite=False):
    """Run every learning rate (resuming finished runs), keep the best, and write <name>_finetuned.pt/.json.

    Prints one line per learning rate; the per-epoch history is kept in the JSON records.
    """
    weights_path, record_path = finetuned_files(name, weights_dir)
    if record_path.is_file() and weights_path.is_file() and not overwrite:
        record = json.loads(record_path.read_text())
        if key(record['recipe']) == key(recipe):
            print(f'{name}: already fine-tuned (lr {record["chosen_learning_rate"]:g}, '
                  f'val {record["val_accuracy"]:.4f}, test {record["test_accuracy"]:.4f}). Skipped.', flush=True)
            return record
        print(f'{name}: saved model used a different recipe. Training again.', flush=True)
    root, parts, manifest_sha = data
    runs_dir = Path(weights_dir) / 'runs'
    runs_dir.mkdir(parents=True, exist_ok=True)
    loader = lambda part, shuffle: make_loader(root, parts[part], recipe['batch_size'], shuffle, recipe['seed'],
                                               workers, device)
    loaders = None
    runs = []
    for lr in recipe['learning_rates']:
        run_state, run_record = runs_dir / f'{name}_lr{lr:g}.pt', runs_dir / f'{name}_lr{lr:g}.json'
        saved = json.loads(run_record.read_text()) if run_record.is_file() and run_state.is_file() else {}
        if saved.get('recipe_id') == recipe_id(recipe) and not overwrite:
            runs.append(saved)
            note = 'finished earlier, reused'
        else:
            loaders = loaders or {'train': loader('train', True), 'val': loader('val', False)}
            state, record = train_run(name, lr, loaders, device, recipe, max_batches)
            torch.save(state, run_state)
            write_json(run_record, record)
            runs.append(record)
            note = f'{record["minutes"]:.1f} min'
        print(f'  {name} lr {lr:g}: best val {runs[-1]["best_val_accuracy"]:.4f} '
              f'at epoch {runs[-1]["best_epoch"]} ({note})', flush=True)

    best = max(runs, key=lambda run: (run['best_val_accuracy'], -run['learning_rate']))
    shutil.copyfile(runs_dir / f'{name}_lr{best["learning_rate"]:g}.pt', weights_path)
    network = fine_tuning_network(name)
    network.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    network.to(device)
    test_loss, test_acc = evaluate(network, loader('test', False), device, device.type == 'cuda', max_batches)
    record = {
        'model': name, 'baseline': 'finetuned', 'start_checkpoint': MODEL_IDS[name],
        'weights_file': weights_path.name, 'weights_sha256': hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        'chosen_learning_rate': best['learning_rate'], 'best_epoch': best['best_epoch'],
        'val_accuracy': best['best_val_accuracy'], 'test_accuracy': test_acc, 'test_loss': test_loss,
        'recipe': recipe, 'smoke_max_batches': max_batches, 'runs': runs,
        'classes': BASE_CLASSES, 'images': {part: len(rows) for part, rows in parts.items()},
        'manifest_sha256': manifest_sha, 'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else str(device),
        'torch': torch.__version__, 'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    write_json(record_path, record)
    print(f'{name}: chose lr {best["learning_rate"]:g} (epoch {best["best_epoch"]}), '
          f'val {best["best_val_accuracy"]:.4f}, test {test_acc:.4f}. Saved {weights_path.name}', flush=True)
    del network
    return record


def fine_tune(models, storage, scratch, manifest_path, device=None, workers=4, smoke=False, overwrite=False,
              recipe_overrides=None):
    """Fine-tune each model in turn. smoke=True runs a few batches into a separate folder to test the code."""
    storage = prepare_storage(storage)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    recipe = {**RECIPE, **(recipe_overrides or {})}
    torch.backends.cudnn.benchmark = False
    max_batches = 2 if smoke else None
    weights_dir = finetuned_weights_dir(storage, smoke)
    print(f'Fine-tuning on {device}: {", ".join(models)} -> {weights_dir}', flush=True)
    data = base_class_data(storage, scratch, manifest_path)
    print('Base-class images:', {part: len(rows) for part, rows in data[1].items()}, flush=True)
    return [fine_tune_model(name, data, weights_dir, device, recipe, workers, max_batches, overwrite) for name in models]
