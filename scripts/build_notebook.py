"""Generate the manifest-only notebook; computation remains in src/eurosat/."""
import argparse
import hashlib
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'EuroSAT_Representation_Benchmark.ipynb'


def canonical_cells(notebook):
    return [(c.cell_type, c.source, {k: v for k, v in c.metadata.items() if k != 'execution'})
            for c in notebook.cells]


def build(*, write=True):
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md('''# EuroSAT: four-class manifest benchmark

Compare ResNet-18, ConvNeXt-Tiny, ViT-B/16 and DINOv2 on the **same fixed four-way episodes**. Six base classes are reserved; four novel classes are evaluated. Backbones stay frozen, and logistic regression is fitted on support embeddings.

The workflow is: **1–2 setup → 3 inspect and validate data → 4 inspect model contracts → 5 execute → 6 inspect each model → 7 compare with ResNet**. Smoke runs verify execution; they are not research conclusions. Start in smoke mode.

Colab needs the project source: copy `src/` into `<shared storage>/code/src/`, or set `EUROSAT_CODE_DIR` to a repository checkout. Copy the committed `splits_manifest.json` into shared storage. Locally, open this notebook from the repository and select its Python environment.
'''),
        md('''## 1. Choose the run

These settings choose hardware, model subset and episode budget. No models or data are loaded yet. Smoke uses the first two development and two test episodes at each shot setting; full uses all 100 development and 600 test episodes. The manifest is always validated in full.
'''),
        code('''import os
import sys
import subprocess
from pathlib import Path

TARGET = os.environ.get('EUROSAT_TARGET', 'colab_gpu')  # local for a local VS Code kernel.
MODE = os.environ.get('EUROSAT_MODE', 'smoke')
MODELS = os.environ.get('EUROSAT_MODELS', 'all')
STORAGE_OVERRIDE = os.environ.get('EUROSAT_STORAGE', '')
CODE_DIR = os.environ.get('EUROSAT_CODE_DIR', '')
MANIFEST_FILE = os.environ.get('EUROSAT_SPLITS', '')
REPORT_RESULTS = os.environ.get('EUROSAT_REPORT_RESULTS', '')
REPORT_ONLY = bool(REPORT_RESULTS)

if TARGET not in {'local', 'colab_gpu'} or MODE not in {'smoke', 'full'}:
    raise ValueError('Choose TARGET local/colab_gpu and MODE smoke/full')
'''),
        md('''## 2. Connect storage and import the implementation

The shared Drive folder stores the verified ZIP, weights, embeddings, validation exports and results. Extracted JPEGs stay on local scratch storage. Keep one writer per shared experiment; atomic files are not a distributed Drive lock.

The source directory contains code only. Package imports below do not load pretrained weights. Source changes require restarting the kernel; changing this notebook does not require copying implementations into cells.
'''),
        code('''import importlib.util

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
    default = parent / 'EuroSATLowData' if (parent / 'EuroSATLowData').exists() else parent / 'EuroSATRepresentationBenchmark'
    STORAGE = Path(STORAGE_OVERRIDE or default)
    if STORAGE_OVERRIDE and not STORAGE.is_dir():
        raise FileNotFoundError('Shared storage is unavailable: check the My Drive shortcut and EUROSAT_STORAGE')
    SCRATCH = Path('/content/eurosat-representation-benchmark')
else:
    current = Path.cwd()
    project = next((p for p in [current, *current.parents] if (p / 'pyproject.toml').exists()), current)
    STORAGE = Path(STORAGE_OVERRIDE or project)
    SCRATCH = STORAGE / 'data/extracted'

STORAGE.mkdir(parents=True, exist_ok=True)
candidates = [Path(CODE_DIR)] if CODE_DIR else [Path.cwd(), *Path.cwd().parents, Path('/content/eurosat-representation-benchmark'), STORAGE / 'code']
source_root = next((p for p in candidates if (p / 'src/eurosat/__init__.py').is_file()), None)
if source_root is None:
    raise FileNotFoundError('Set EUROSAT_CODE_DIR to the repository, or copy src/ into <storage>/code/src/')
sys.path.insert(0, str(source_root / 'src'))

print('Persistent storage:', STORAGE)
print('Local extracted images:', SCRATCH)
print('Project source:', source_root)
'''),
        code('''import json
import pandas as pd
from IPython.display import display, Image as NotebookImage

from eurosat.config import Config, MODEL_IDS
from eurosat.experiments import manifest_config, run_manifest_benchmark
from eurosat.models import EMBEDDING_DIMENSIONS
from eurosat.preprocessing import load_source, validate_manifest, select_episodes
from eurosat.preprocessing.audit import episode_table, usage_table, inspect_batch, data_figures
from eurosat.reporting import create_report, display_model_evaluation, baseline_comparison
from eurosat.evaluation import summarize
from eurosat.runtime import check_device, environment

config = manifest_config(MODE, MODELS)
if os.environ.get('EUROSAT_TRIALS'):
    config.trials = int(os.environ['EUROSAT_TRIALS'])
if os.environ.get('EUROSAT_POLICIES'):
    config.policies = os.environ['EUROSAT_POLICIES'].split(',')
if os.environ.get('EUROSAT_VIEWS'):
    config.views = int(os.environ['EUROSAT_VIEWS'])
if REPORT_ONLY:
    saved = json.loads(Path(REPORT_RESULTS).read_text())
    if 'episode_source' not in saved['manifest']:
        raise ValueError('This notebook expects four-class manifest results')
    config = Config(**saved['manifest']['config'])
config.validate()

manifest_path = Path(MANIFEST_FILE) if MANIFEST_FILE else STORAGE / 'splits_manifest.json'
device = None if REPORT_ONLY else check_device(TARGET)
print('Run:', config.mode, '| models:', config.models, '| shots:', config.shots)
print('Test episodes per shot:', config.trials, '| policies:', config.policies)
print('Software:', environment())
'''),
        md('''## 3. Data processing: inspect each stage

No split is generated here. The committed manifest is the contract for which images may enter development and final evaluation. The following stages expose the source inventory, all-pool checks and selected episode assignments before inference starts.

### 3.1 Verify the source archive and local images

`load_source()` verifies the ZIP checksum, checks each extracted file against its ZIP CRC, and restores missing/changed local files from that ZIP. The cached decoded-RGB audit checks image decoding and exact pixel duplicates; conflicting labels or undecodable images stop the run. The table distinguishes all ten source classes from the four evaluated classes.
'''),
        code('''if REPORT_ONLY:
    print('Report only: source acquisition is skipped.')
else:
    source = load_source(STORAGE, SCRATCH, config.seed)

    display(pd.DataFrame(source.inventory))
    display(pd.Series(source.integrity, name='source_check').to_frame())
    print('Native image properties:', source.properties)
    print('Source audit:', source.audit_path)
'''),
        md('''### 3.2 Validate the entire shared manifest

`validate_manifest()` checks complete file coverage, unique class-pool membership, development/test isolation by filenames **and decoded pixels**, and correct support/query sizes and separation in **every committed episode**. Smoke mode does not weaken these checks.

The six base classes are reserved. Novel development labels are extra selection information outside the support-shot budget. Novel test queries never choose C, augmentation settings or model checkpoints.
'''),
        code('''if not REPORT_ONLY:
    validated = validate_manifest(manifest_path, source)

    print('Manifest SHA256:', validated.sha256)
    display(pd.DataFrame(validated.pools))
    display(pd.DataFrame(validated.episode_counts))
    display(pd.DataFrame(validated.checks))
'''),
        md('''### 3.3 Select the run's fixed episodes and export their lineage

`select_episodes()` takes the first configured episodes without shuffling or resampling. It resolves filenames to stable image indices and saves the source checks, manifest fingerprint, exact record list and exact selected episode indices under `cache/data-audits/`.

The usage table makes repeated image use visible. Query images can recur across episodes; these are repeated observations, not additional independently labeled images. The example table shows the actual 5-shot development support/query assignments.
'''),
        code('''if not REPORT_ONLY:
    prepared = select_episodes(source, validated, config, STORAGE)

    display(usage_table(prepared))
    display(episode_table(prepared, phase='dev', shots=5, episode=0))
    print('Unique images selected for this run:', len(prepared.records))
    print('Inspect exported records, episodes and checks:', prepared.audit_dir)
'''),
        md('''### 3.4 Inspect model inputs and development support examples

The same Dataset converts images to RGB, resizes bilinearly to 224×224 and applies ImageNet channel normalization. Development/query reads are deterministic. The check prints tensor shape, dtype, finite-value status and repeated-read equality. Normalized values need not lie between zero and one.

The montage uses development **support** images from episode 0, never test images or high-scoring examples. The pool chart shows the manifest's actual development/test allocation.
'''),
        code('''if not REPORT_ONLY:
    display(pd.Series(inspect_batch(prepared, config), name='input_check').to_frame())

    data_plots = data_figures(prepared, validated)
    for filename in data_plots.values():
        display(NotebookImage(filename=filename, width=850))
'''),
        md('''## 4. Model implementation map

Each teammate edits one adapter in `src/eurosat/models/`. `build()` must return a pretrained network with its classification head removed. The shared loader freezes weights and sets evaluation mode; the shared feature extractor checks embedding width and finite values.

Do not add private splitting, preprocessing or classifier code inside an adapter. ResNet-18 is the baseline. Updating weights or extraction semantics requires updating the checkpoint/cache identity before reusing embeddings.
'''),
        code('''display(pd.DataFrame([{'model': name,
                       'implementation': f'src/eurosat/models/{name}.py',
                       'embedding_width': EMBEDDING_DIMENSIONS[name],
                       'checkpoint': MODEL_IDS[name],
                       'included': name in config.models}
                      for name in MODEL_IDS]))
'''),
        md('''## 5. Run or resume the four-model evaluation

This call consumes the already inspected `prepared` episodes. Models execute sequentially; each is released before the next is loaded. Matching chunks and completed trials are reused. Missing classifiers use development episodes to choose C, then fit each test episode's support set and predict only its query set. No backbone training occurs.

The shared classifier is L2-regularized logistic regression on L2-normalized embeddings, with C in `[0.1, 1, 10]`. Augmentation, if enabled, is support-only with fixed strengths. JSON/CSV, numeric classifier checkpoints and timings are saved under the printed experiment directory.
'''),
        code('''if REPORT_ONLY:
    result_path = Path(REPORT_RESULTS)
    results = {'output': str(result_path.parent), 'summary': summarize(saved['trials']),
               'report': create_report(result_path, os.environ.get('EUROSAT_REPORT_OUTPUT'))}
else:
    results = run_manifest_benchmark(config, STORAGE, SCRATCH, manifest_path,
                                    target=TARGET, episode_limit=config.trials, prepared=prepared)

print('Saved model results:', results['output'])
'''),
        md('''## 6. Inspect each model

These cells read saved predictions and metrics; they do not rerun inference. Each shows mean accuracy/Macro-F1, development-selected C, episode count, per-class recall and a normalized confusion matrix at the largest shot budget without augmentation. A model excluded from `MODELS` is explicitly marked as not run. Smoke scores are execution checks only.
'''),
    ]
    for title, name, description in [
        ('6.1 ResNet-18 — baseline', 'resnet18', 'A standard CNN. Use its paired results as the reference for the other three backbones.'),
        ('6.2 ConvNeXt-Tiny', 'convnext_tiny', 'A modern CNN. Compare it with ResNet on the same images and classifier procedure.'),
        ('6.3 ViT-B/16', 'vit_b_16', 'A supervised Vision Transformer. Its adapter returns the final class-token embedding.'),
        ('6.4 DINOv2 ViT-S/14', 'dinov2_vits14', 'A self-supervised representation. Differences also include pretraining data and model size, not only the learning objective.'),
    ]:
        cells.extend([md(f'### {title}\n\n{description}'), code(f"display_model_evaluation(results, '{name}')\n")])
    cells.extend([
        md('''## 7. Compare all models with the ResNet baseline

Accuracy/Macro-F1 are fractions. Differences from ResNet are **paired percentage points** using matching policy, shot and episode IDs. ResNet rows have zero paired gain by definition. If ResNet was not selected, this table shows scores only and does not claim a baseline gain.

Images recur across episodes, so variability and bootstrap intervals describe this fixed sampling schedule; they do not establish geographic generalization. Review negative as well as positive effects. Full results require the deliberate full episode budget.
'''),
        code('''display(baseline_comparison(results).round(4))

report = results['report']
for name in ['model_by_shot', 'augmentation_gains']:
    display(NotebookImage(filename=report['figures'][name], width=1000))

print('Combined figures, class metrics, timing tables and provenance:', report['output'])
'''),
    ])
    for i, cell in enumerate(cells):
        cell.id = 'manifest-' + hashlib.sha256(f'{i}:{cell.source}'.encode()).hexdigest()[:12]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU', 'colab': {'gpuType': 'T4', 'provenance': []}})
    if NOTEBOOK.exists():
        old = [c for c in nbf.read(NOTEBOOK, 4).cells if c.cell_type == 'code']
        new = [c for c in notebook.cells if c.cell_type == 'code']
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
    notebook = build(write=not args.check)
    if args.check and canonical_cells(notebook) != canonical_cells(nbf.read(NOTEBOOK, 4)):
        raise SystemExit('Notebook is stale; run python scripts/build_notebook.py')
    print(NOTEBOOK)


if __name__ == '__main__':
    main()
