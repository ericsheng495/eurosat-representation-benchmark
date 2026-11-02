"""Build the separate, readable Colab/VS Code fine-tuning notebook."""
import argparse
import ast
import hashlib
from pathlib import Path
import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_notebook import canonical_cells, compact_source_blocks

NOTEBOOK = ROOT / 'EuroSAT_Fine_Tuning.ipynb'
MODULES = ['config', 'cache', 'runtime', 'augment', 'data', 'models',
           'features', 'evaluation', 'exploration', 'finetuning', 'finetuning_reporting']
SELECTIONS = {
    'cache': ['key', 'digest_file', 'atomic', 'write_json', 'prepare_storage'],
    'runtime': ['environment', 'check_device'],
    'features': ['Images'],
    'evaluation': ['score'],
    'exploration': ['STYLE', 'save_figure'],
}


def implementation_blocks():
    blocks = []
    for name, source in compact_source_blocks(MODULES, SELECTIONS, preimported={'os', 'sys', 'subprocess', 'Path'}):
        if name != 'finetuning':
            blocks.append((name, source))
            continue
        # Separate settings, model, training and orchestration without copying definitions.
        markers = {'AdaptedClassifier': 'partial_model', 'optimizer_for': 'training_and_recovery',
                   'run_finetuning': 'experiment_runner'}
        boundaries = [(0, 'fine_tuning_settings')]
        boundaries.extend((node.lineno - 1, markers[node.name]) for node in ast.parse(source).body
                          if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in markers)
        lines = source.splitlines()
        for i, (start, label) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines)
            blocks.append((label, '\n'.join(lines[start:end]).strip()))
    return blocks


def build(*, write=True):
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md('''# EuroSAT: DINOv2 Partial Fine-Tuning

Optional supervised adaptation of **DINOv2 ViT-S/14**, separate from `EuroSAT_Representation_Benchmark.ipynb`. Update the **last Transformer block, final LayerNorm and ten-class head**; freeze the patch embedding, positional/class tokens and earlier blocks.

We compare the same pretrained model and neural classification head under **frozen-backbone** and **last-block fine-tuning** conditions. Both receive identical original supports, fixed augmentation views, head initialization and batch order. This control isolates the adaptation effect more directly than comparing a neural head against the main notebook's logistic regression.

The proposal treats fine-tuning as an optional separate experiment. All ten classes, original 60/20/20 splits and nested 1/5/10-shot supports are reused. Development labels choose the epoch; held-out labels are used only after training. Saved smoke outputs validate execution and are not final findings.
'''), md('''## Research direction: how much should we adapt?

**Research question:** can adapting DINOv2's pretrained representation improve few-shot satellite-image classification more than augmenting support images alone? The reported small augmentation gains and larger gains from additional originals motivate this question; they do not establish that fine-tuning will help.

| Direction | What changes | Decision |
|---|---|---|
| Partial fine-tuning | Last 1 block + final LayerNorm + head | Priority 1; implemented and selected |
| Two-block ablation | Last 2 blocks + final LayerNorm + head | Implemented; optional sensitivity study |
| LoRA / adapters | Small additional trainable modules | Future comparison; not implemented |
| Normalization-only adaptation | Final LayerNorm + head | Future smaller-scope control; not implemented |

**Why one block first?** The hypothesis is that late representations can adapt to satellite imagery while limiting updated weights. This is a conservative scope relative to two blocks, not a proven optimum. The head-only baseline uses the same initialization and optimizer. Full-backbone training is disabled.

**Decision rule:** fix the one-block protocol before the final study. Set `LAST_BLOCKS = 2` only for a separately reported ablation; do not choose the better scope from query scores. Learning rates are starting settings. Report negative or inconsistent gains, especially at 1 shot.

Background: [DINOv2 paper](https://arxiv.org/abs/2304.07193) and [official implementation](https://github.com/facebookresearch/dinov2). Neither establishes that this EuroSAT adaptation will outperform frozen features.

### Alignment with the project proposal

The [proposal](docs/project-proposal.pdf), **page 4**, explicitly permits fine-tuning as an optional extension reported separately from frozen experiments. The primary benchmark remains the four frozen representations with common logistic regression.

| Proposal commitment | This extension |
|---|---|
| DINOv2 ViT-S/14; exact checkpoint (p. 1) | Same official pinned pretrained backbone |
| Ten classes, deduplication, 60/20/20 split, nested 1/5/10 shots (p. 3–4) | Same manifests and original support samples |
| Support-only augmentation and repeated-original control (p. 4–5) | None / repeat / flip crossed with frozen / adapted backbone |
| Original + four views versus five originals (p. 5) | Paired 1+4 versus 5-original contrast in full mode |
| Accuracy, Macro-F1, class recall, confusion matrices and cost (p. 5) | Classification metrics, parameter counts and training time |

This **2 × 3 extension** runs two adaptation conditions × three support treatments × three shot budgets × 20 full trials = **360 fits**. Flip is a prespecified proposal-listed policy, not a query-selected winner. Repeat and flip both add four rows per original in full mode; only flip changes pixels. The primary notebook retains the broader augmentation and nine-view experiments.

The neural-head baseline differs from the primary logistic-regression procedure. Within this notebook, the head, optimizer and original supports are matched to isolate adaptation. Cross-notebook differences cannot be attributed solely to fine-tuning. This follow-up is exploratory because existing query results motivated it; it is not a new independent confirmation set.
'''), md('## 1. Runtime and model'), code('''import os
import sys
import subprocess
from pathlib import Path

TARGET = os.environ.get('EUROSAT_TARGET', 'colab_gpu')  # Set local for a local VS Code kernel.
MODE = os.environ.get('EUROSAT_FINETUNE_MODE', 'smoke')
MODEL = 'dinov2_vits14'  # This extension focuses on DINOv2.
LAST_BLOCKS = int(os.environ.get('EUROSAT_FINETUNE_BLOCKS', '1'))  # Optional ablation: 2.
if TARGET not in {'colab_gpu', 'local'} or MODE not in {'smoke', 'full'}:
    raise ValueError('Choose TARGET colab_gpu/local and MODE smoke/full')
print('Selected model:', MODEL, '| mode:', MODE)
'''), md('## 2. Dependencies and shared storage'), code('''import importlib.util

requirements = {'torch': 'torch>=2.6,<3', 'torchvision': 'torchvision>=0.21,<1',
                'numpy': 'numpy>=1.26,<3', 'sklearn': 'scikit-learn>=1.4,<1.8',
                'PIL': 'Pillow>=10,<13', 'pandas': 'pandas>=2.2,<3',
                'matplotlib': 'matplotlib>=3.8,<4', 'seaborn': 'seaborn>=0.13,<0.14'}
missing = [package for module, package in requirements.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', *missing])
if TARGET == 'colab_gpu':
    from google.colab import drive
    drive.mount('/content/drive')
    parent = Path('/content/drive/MyDrive/ColabNotebooks/cs7643')
    previous = parent / 'EuroSATLowData'
    default = previous if previous.exists() else parent / 'EuroSATRepresentationBenchmark'
    STORAGE = Path(os.environ.get('EUROSAT_STORAGE', str(default)))
    SCRATCH = Path('/content/eurosat-representation-benchmark')
else:
    current = Path.cwd()
    project = next((p for p in [current, *current.parents] if (p / 'pyproject.toml').exists()), current)
    STORAGE = Path(os.environ.get('EUROSAT_STORAGE', str(project)))
    SCRATCH = STORAGE / 'data/extracted'
STORAGE.mkdir(parents=True, exist_ok=True)
print('Shared dataset/weights:', STORAGE)
print('Separate adaptation checkpoints: cache/finetuning; results: outputs/finetuning')
'''), md('''## 3. Actual implementation

Run the helper cells in order. Imports appear once; only the required shared helpers are included. All code is visible and generated from `src/eurosat/`, so fixes have one source of truth. Training lives in `finetuning.py`; tables and plots live in `finetuning_reporting.py`. Rebuild with `python scripts/build_finetuning_notebook.py` after edits.
''')]
    for name, source in implementation_blocks():
        cells.extend([md('### ' + name.replace('_', ' ').capitalize()),
                      code(source, metadata={'tags': ['finetuning_implementation', name]})])
    cells.extend([md('''## 4. Training protocol

`last_block` trains only the final backbone stage/block and a fresh ten-class neural head. The matched `frozen_head` control learns the same head with a frozen backbone. Both use cross-entropy and AdamW, with a smaller backbone learning rate (head: 0.001; backbone: 0.00001).

Pretrained BatchNorm statistics and backbone dropout remain in evaluation mode during adaptation, which is useful for tiny support batches. Gradients still update the selected parameters. Development accuracy selects the epoch; development loss breaks ties. The query set never chooses an epoch. Original and augmented supports stay within their original partition.

Smoke: 40 images/class, 1 trial, 2 epochs and 1 added view. `MODE = 'full'`: entire dataset, 20 trials, up to 10 epochs and 4 added views. **Mode changes the data/trial budget, never which layers are unfrozen.** The larger budget evaluates the development pool every epoch and is substantially more expensive than frozen-feature extraction.
'''), code('''ft_config = FineTuneConfig.full(MODEL) if MODE == 'full' else FineTuneConfig(model=MODEL)
ft_config.last_blocks = LAST_BLOCKS
ft_config.validate()
device = check_device(TARGET)
print(json.dumps(asdict(ft_config), indent=2))
print('Software:', environment())
'''), md('## 5. Verify the data and original-shot budgets'), code('''from IPython.display import display, Image as NotebookImage, Markdown

image_root, records, split, dataset_id = prepare_data(prepare_storage(STORAGE), SCRATCH, ft_config.data_config())
supports = make_trials(records, split, ft_config.data_config())
display(pd.DataFrame([{'class': name, **{part: sum(records[i]['label'] == label for i in ids)
        for part, ids in split.items()}} for label, name in enumerate(CLASSES)]))
print('Dataset identity:', dataset_id)
print('Trial 0 original supports:', {k: len(ids) for k, ids in supports[0].items()})
example = records[split['support'][0]]
display(NotebookImage(filename=str(image_root / example['path']), width=224))
print('Unaugmented support example:', CLASSES[example['label']])
'''), md('''## 6. Fine-tune or resume

Each epoch saves changed tensors, optimizer state, best development weights and history atomically. A restart can repeat an incomplete epoch; completed epochs and jobs are reused. Checkpoint keys include the dataset, model, optimizer settings, budgets and software. Backbone embeddings cannot be reused during weight updates; the trained weights are the cacheable artifact.

Training checkpoints are in their own namespace and use tensor-only loading. Test evaluation happens after the best epoch is restored. This cell runs DINOv2 with its matched head-only control under none/repeat/flip support treatments.
'''), code('''fine_tuning_results = run_finetuning_study(ft_config, STORAGE, SCRATCH, target=TARGET)
'''), md('''## 7. Actual results and learning curves

The plots show accuracy/Macro-F1, learning curves and a prespecified no-augmentation confusion comparison at the largest shot budget in trial 0. These matrices describe that trial only, not a selected best trial. Curves show mean ± trial standard deviation; a single smoke trial has no between-trial uncertainty estimate. Report these results separately from the frozen-logistic-regression benchmark. Compare adaptation against this notebook's matched neural-head control first. The full metrics, histories and settings are saved as JSON; summary/trial tables and PNG/PDF figures accompany them.
'''), code('''adaptation_report = finetuning_figures(fine_tuning_results)
display(adaptation_report['summary'].round(3))
for name, filename in adaptation_report['figures'].items():
    display(Markdown('**' + name.replace('_', ' ').capitalize() + '**'))
    display(NotebookImage(filename=filename, width=900))
display(adaptation_report['table'][['strategy', 'augmentation', 'shots', 'selected_epoch',
                                    'trainable_parameters', 'total_parameters']])
display(adaptation_report['class_recall_summary'].round(3))
display(adaptation_report['contrast_summary'].round(3))
print('Artifacts:', fine_tuning_results['output'])
print('Smoke results verify execution only; positive paired gains favor the named candidate.')
'''), md('''## 8. Extension checklist

- [x] Keep the adaptation experiment separate from the primary frozen-backbone study.
- [x] Reuse shared partitions, nested original supports and development-only epoch selection.
- [x] Match frozen-head and adaptation head initialization, training inputs and optimizer rules.
- [x] Persist epoch checkpoints and restore the selected weights before query evaluation.
- [x] Select DINOv2 last-block adaptation and define paired none/repeat/flip comparisons.
- [ ] Lock learning rates, epoch budget and any two-block ablation before final query evaluation.
- [ ] Run the agreed full grid on the intended GPU and review class-specific errors.
- [ ] Report extra development labels, training cost and differences from the primary logistic-regression protocol.
''')])
    for i, cell in enumerate(cells):
        cell.id = 'ft-' + hashlib.sha256(f'{i}:{cell.source}'.encode()).hexdigest()[:12]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU', 'colab': {'gpuType': 'T4', 'provenance': []}})
    if NOTEBOOK.exists():
        previous = nbf.read(NOTEBOOK, as_version=4)
        old = [c for c in previous.cells if c.cell_type == 'code']
        new = [c for c in cells if c.cell_type == 'code']
        if [c.source for c in old] == [c.source for c in new]:
            for before, after in zip(old, new):
                after.outputs, after.execution_count = before.outputs, before.execution_count
    nbf.validate(notebook)
    if write:
        nbf.write(notebook, NOTEBOOK)
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    generated = build(write=not args.check)
    if args.check and (not NOTEBOOK.exists() or canonical_cells(nbf.read(NOTEBOOK, 4)) != canonical_cells(generated)):
        raise SystemExit('Fine-tuning notebook is stale; run python scripts/build_finetuning_notebook.py')
    print(NOTEBOOK)


if __name__ == '__main__':
    main()
