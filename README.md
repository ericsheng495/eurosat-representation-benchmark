# EuroSAT Representation Benchmark

CS7643 project comparing CNN and Transformer representations for satellite-image classification with 1, 5 and 10 labeled originals per class.

**Team:** Andrew Dao, Eric Sheng, Eunhyeok Jang and Yanru Yang.

The [project proposal](docs/project-proposal.pdf) matches `Project Proposal_draft2 (3).pdf`, downloaded November 8, 2026. The current implementation uses the shared **four-class manifest protocol**. The proposal describes a ten-class 60/20/20 image-split study; that workflow is currently removed. Treat these as different evaluation tasks.

## Repository structure

```text
EuroSAT_Representation_Benchmark.ipynb  One notebook: data inspection → models → comparisons
splits_manifest.json                   Committed pools and fixed episode assignments
make_splits.py                         Manifest generator and standalone loaders
src/eurosat/
  preprocessing/
    data.py                           Download, checksum/CRC verification and pixel audit
    manifest.py                       Validate pools/episodes and select exact images
    audit.py                          Inspection tables, input checks and dataset figures
    transforms.py                     Resize, normalization and deterministic augmentation
    images.py                         Shared PyTorch Dataset
  models/
    resnet18.py                        Standard CNN baseline
    convnext_tiny.py                   Modern CNN
    vit_b_16.py                        Supervised Vision Transformer
    dinov2_vits14.py                   Self-supervised Vision Transformer
    __init__.py                       Shared frozen-backbone loader
  config.py, runtime.py                Presets, checkpoint IDs and hardware checks
  cache.py, features.py                Atomic checkpoints and chunked embeddings
  evaluation.py                       Shared logistic regression and metrics
  experiments.py, pipeline.py          Run configuration and sequential model execution
  reporting.py, cli.py                 Saved-result tables, plots and run/report commands
scripts/
  build_notebook.py                    Canonical notebook cells and Markdown
  execute_notebook.py                  Fresh-kernel local execution
requirements/local.txt                Local installation
docs/                                 Proposal, methods, metrics and team instructions
SPLITS.md                             Manifest definitions and label ordering
TODO.md                               Checkoff plan and remaining experiments
```

## Notebook workflow

Each section opens with three lines: what it **does**, what **you get**, and what to **edit**. Only the Settings cell needs editing; section helpers live in `src/eurosat/notebook.py`.

| Section | What it does |
|---|---|
| 1 · Settings | The only cell to edit: `TARGET`, `MODE`, `MODELS`, `POLICIES`, `VIEWS`, `AUGMENTATION`, `SHOW_DETAILS`, `REPORT_RESULTS` |
| 2 · Setup | Installs packages, connects storage, finds `src/` (visible code), then validates the settings and prints a run summary |
| 3 · Data check | Verifies the archive and every manifest pool/episode, selects the fixed episodes, shows a pass/fail summary |
| 4 · Run | Runs or resumes each model; loads saved results in report-only mode |
| 5 · Results: models | No-augmentation scores, paired gains over ResNet-18, per-model recall and confusion matrix |
| 6 · Results: augmentation | Paired gains per method, development-selected method, per-class effects |
| 7 · Saved files | Output folders |

`SHOW_DETAILS = True` adds every data-check table (source inventory, pools, image reuse, example assignments, input check) and extra result figures.

### Data checks

Section 3 verifies the EuroSAT ZIP checksum and extracted-file CRCs (restoring missing or changed files), audits decoding and exact-pixel duplicates, validates every committed pool and episode, selects fixed episodes without resampling, and checks normalized `3 × 224 × 224` inputs. Records, assignments and check results are exported.

Six base classes are reserved and do not train these backbones. Four novel classes—HerbaceousVegetation, Pasture, PermanentCrop and River—are evaluated. Each episode has k support originals and 15 queries per class. Development/test pools are disjoint; images may recur across episodes. Supports are not guaranteed nested across shot budgets. See [SPLITS.md](SPLITS.md) and [dataset checks](docs/dataset.md).

### Four frozen models and shared execution

| Model | Role | Embedding width | Implementation |
|---|---|---:|---|
| ResNet-18 | CNN baseline | 512 | `models/resnet18.py` |
| ConvNeXt-Tiny | Modern CNN | 768 | `models/convnext_tiny.py` |
| ViT-B/16 | Supervised Transformer | 768 | `models/vit_b_16.py` |
| DINOv2 ViT-S/14 | Self-supervised Transformer | 384 | `models/dinov2_vits14.py` |

Implementation paths are relative to `src/eurosat/`. Every model receives identical episode assignments and preprocessing. Backbones remain frozen in evaluation mode; embeddings are L2-normalized. Development episodes select logistic-regression C from `[0.1, 1, 10]`. A classifier is then fitted to each test episode's supports and evaluated on its queries. Query scores never choose settings.

Models run sequentially and release GPU memory between backbones. The default policy is no augmentation. Augmentation conditions, view count and magnitudes are set in the notebook's Settings cell (`POLICIES`, `VIEWS`, `AUGMENTATION`); defaults live in `AUGMENTATION_DEFAULTS` in `src/eurosat/config.py`. Optional deterministic support-only policies and repeated-original controls remain available; the default run does not execute the proposal's complete augmentation/view-budget study. See [methods](docs/methodology.md).

### Results

Section 5 compares all selected models with the ResNet-18 baseline using paired episode differences in percentage points, then shows each model's accuracy, Macro-F1, development-selected C, class recall and confusion matrix. Section 6 reports augmentation gains when `POLICIES` includes methods other than `none`.

Results include `results.json`, `summary.csv`, class metrics, timing tables, provenance and report plots. Smoke scores validate execution only. Image reuse and pretraining differences limit interpretation; see [metrics](docs/metrics.md).

## Run in Google Colab

Open [EuroSAT_Representation_Benchmark.ipynb](EuroSAT_Representation_Benchmark.ipynb) with a GPU runtime. Set `CODE_DIR` to a repository checkout, or copy `src/` into `<shared storage>/code/src/`. Copy the committed `splits_manifest.json` into shared storage, or set `MANIFEST_FILE` to its path. Uploading the notebook alone does not include these dependencies.

Edit the Settings cell (Section 1), then run cells in order:

```python
MODE = 'smoke'  # full: 100 development and 600 test episodes per shot
MODELS = 'all'  # or resnet18, convnext_tiny, vit_b_16, dinov2_vits14
```

Smoke uses two development and two test episodes for each of 1, 5 and 10 shots. Model subsets can be comma-separated. For automated runs, environment variables override the Settings cell: `EUROSAT_TARGET`, `EUROSAT_MODE`, `EUROSAT_MODELS`, `EUROSAT_POLICIES`, `EUROSAT_VIEWS`, `EUROSAT_TRIALS`, `EUROSAT_STORAGE`, `EUROSAT_CODE_DIR`, `EUROSAT_SPLITS` and `EUROSAT_REPORT_RESULTS`.

## Team Collaboration and Shared Google Drive Cache

Create a Google Drive folder, share it with teammates as **Editors**, and have each person add a shortcut to it in **My Drive**. Set `STORAGE_DIR` in the notebook's Settings cell or configure:

```python
import os

os.environ['EUROSAT_STORAGE'] = '/content/drive/MyDrive/EuroSATRepresentationBenchmark'
```

Use each member's actual shortcut path. Without an override, the notebook reuses the older personal `MyDrive/ColabNotebooks/cs7643/EuroSATLowData` folder if present, otherwise `EuroSATRepresentationBenchmark`. A personal folder is not automatically shared.

```text
<shared storage>/
  splits_manifest.json      Fixed episode definitions
  code/src/                 Source package, if no repository checkout
  data/EuroSAT.zip          Verified archive
  cache/data-audits/        Checks, selected records/episodes and dataset figures
  cache/weights/            Pretrained weights
  cache/features/           Original and augmented embeddings
  cache/development/        Existing development-selection cache location
  cache/experiments/        Trial manifests, classifier selections and checkpoints
  outputs/<experiment-id>/  results.json, summary.csv and report figures
```

Extracted JPEGs stay in local Colab scratch storage and are restored from the ZIP after a reset. Matching features and completed trials are reused with `CACHE HIT` / `CACHE MISS` logs. Identities include dataset/checkpoint, preprocessing, augmentation settings and selected images/views.

Use one writer per shared experiment at a time: atomic file publication does not provide a distributed Google Drive lock. GitHub stores source and the manifest; Drive stores datasets, weights, embeddings and generated artifacts.

To regenerate reports without a GPU or data download, set `REPORT_RESULTS` to a saved `results.json` in the notebook.

## Local setup and contributions

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/local.txt
python -m eurosat run --mode smoke --models all --manifest splits_manifest.json
python -m eurosat run --mode smoke --models dinov2_vits14 --manifest splits_manifest.json
python -m eurosat report /path/to/results.json
```

In VS Code, select `.venv` as the kernel and set `TARGET = 'local'`. Each teammate edits their model adapter while sharing preprocessing and evaluation. Restart the kernel after changing imported Python code.

Edit notebook cells and Markdown in the builder, then regenerate and inspect a small run:

```sh
python scripts/build_notebook.py
python scripts/execute_notebook.py --mode smoke --models all --output work/smoke.ipynb
```

Use feature branches and pull requests. Keep the committed manifest shared; do not regenerate separate splits for each teammate. See [collaboration instructions](docs/collaboration.md) and [remaining work](TODO.md).
