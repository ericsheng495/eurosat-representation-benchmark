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
  training.py                         Optional DINOv2 base training and epoch resume
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

### 1–2. Configuration, environment and storage

Choose smoke/full mode, all models or a subset, and local/Colab hardware. Connect persistent storage and import the shared source package. These steps do not train or load a backbone.

### 3. Data processing and inspection

The notebook exposes four stages instead of hiding preparation in one call:

1. Verify the EuroSAT ZIP checksum and extracted-file CRCs, restore missing/changed files, and audit decoding and exact-pixel duplicates. Display all ten source classes and native image properties.
2. Validate every committed pool and episode, including filename coverage, class membership, shot/query counts and development/test isolation.
3. Select fixed episodes without resampling. Export the exact image records, episode assignments and validation results, and display image-reuse counts and example paths.
4. Inspect native development support images, pool distributions and normalized `3 × 224 × 224` inputs. Check finite values and deterministic reads.

In the default pretrained mode, six base classes are reserved. Optional scratch mode uses their train/validation partitions; base test images stay untouched. Four novel classes—HerbaceousVegetation, Pasture, PermanentCrop and River—are evaluated. Each episode has k support originals and 15 queries per class. Development/test pools are disjoint; images may recur across episodes. Supports are not guaranteed nested across shot budgets. See [SPLITS.md](SPLITS.md) and [dataset checks](docs/dataset.md).

### 4–5. Four frozen models and shared execution

| Model | Role | Embedding width | Implementation |
|---|---|---:|---|
| ResNet-18 | CNN baseline | 512 | `models/resnet18.py` |
| ConvNeXt-Tiny | Modern CNN | 768 | `models/convnext_tiny.py` |
| ViT-B/16 | Supervised Transformer | 768 | `models/vit_b_16.py` |
| DINOv2 ViT-S/14 | Self-supervised Transformer | 384 | `models/dinov2_vits14.py` |

Implementation paths are relative to `src/eurosat/`. Every model receives identical episode assignments and preprocessing. Backbones remain frozen in evaluation mode; embeddings are L2-normalized. Development episodes select logistic-regression C from `[0.1, 1, 10]`. A classifier is then fitted to each test episode's supports and evaluated on its queries. Query scores never choose settings.

Models run sequentially and release GPU memory between backbones. The default policy is no augmentation. Optional deterministic support-only policies and repeated-original controls remain available; the default run does not execute the proposal's complete augmentation/view-budget study. See [methods](docs/methodology.md).

### Optional DINOv2 scratch-training toggle

In the notebook configuration:

```python
MODELS = 'dinov2_vits14'       # Or all to retain three pretrained comparisons.
DINO_INITIALIZATION = 'scratch'    # pretrained remains the default.
SCRATCH_TRAINING = {}         # Use mode defaults; override below when needed.
```

Scratch mode initializes DINOv2 randomly, trains all its backbone parameters plus a temporary six-class head on **base train**, selects the best epoch on **base validation**, then removes the head and freezes the backbone. The toggle affects DINOv2 only; ResNet-18, ConvNeXt-Tiny and ViT-B/16 always use their original pretrained weights. The existing four-class episodes and logistic-regression evaluation are unchanged. Neither base test nor novel images train or select the backbone.

This means supervised training of the ViT-S/14 architecture, **not** reproducing the DINOv2 self-supervised training method. Compare its results separately from the pretrained representation study.

Full defaults: 20 epochs, batch size 8, AdamW at `3e-4`, weight decay `0.05`, and all 10,200 base-training/3,400 base-validation images. Smoke uses one epoch and two images per base class in each partition. These defaults are an executable starting point, not a tuned recipe or matched pretraining budget. Use `SCRATCH_TRAINING = {'epochs': 50, 'learning_rate': 0.0001}` to override settings; `per_class_limit` can limit images for debugging.

`EUROSAT_DINO_INITIALIZATION` and JSON `EUROSAT_SCRATCH_TRAINING` provide equivalent environment controls. Weights, optimizer state and the best backbone are saved under `cache/weights/base-training/<plan-id>/`. Resume starts after the last completed epoch; an interrupted epoch may repeat. Epoch seeds make data order/flips repeatable on the same stack, but do not guarantee bitwise equality across devices. Metadata records the training pools, seed, settings and best epoch.

Scratch features/results use the selected weights' SHA256 identity and cannot collide with pretrained artifacts. Shared Drive writers should still be serialized. Existing pretrained cache identities remain unchanged.

```sh
python -m eurosat run --mode smoke --models dinov2_vits14 --dino-initialization scratch --manifest splits_manifest.json
python -m eurosat run --mode full --models dinov2_vits14 --dino-initialization scratch --scratch-epochs 50 --manifest splits_manifest.json
```

### 6–7. Individual evaluation and model comparisons

Each model has its own result cell with accuracy, Macro-F1, development-selected C, class recall and a confusion matrix. The final cells compare all selected models with the ResNet-18 baseline using paired episode differences in percentage points.

Results include `results.json`, `summary.csv`, class metrics, timing tables, provenance and report plots. Smoke scores validate execution only. Image reuse and pretraining differences limit interpretation; see [metrics](docs/metrics.md).

## Run in Google Colab

Open [EuroSAT_Representation_Benchmark.ipynb](EuroSAT_Representation_Benchmark.ipynb) with a GPU runtime. Set `EUROSAT_CODE_DIR` to a repository checkout, or copy `src/` into `<shared storage>/code/src/`. Copy the committed `splits_manifest.json` into shared storage, or set `MANIFEST_FILE` to its path. Uploading the notebook alone does not include these dependencies.

Set the first configuration cell, then run cells in order:

```python
MODE = 'smoke'  # full: 100 development and 600 test episodes per shot
MODELS = 'all'  # or resnet18, convnext_tiny, vit_b_16, dinov2_vits14
```

Smoke uses two development and two test episodes for each of 1, 5 and 10 shots. Model subsets can be comma-separated. Environment overrides include `EUROSAT_MODE`, `EUROSAT_MODELS`, `EUROSAT_STORAGE`, `EUROSAT_CODE_DIR` and `EUROSAT_SPLITS`.

## Team Collaboration and Shared Google Drive Cache

Create a Google Drive folder, share it with teammates as **Editors**, and have each person add a shortcut to it in **My Drive**. Set `STORAGE_OVERRIDE` in the notebook or configure:

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
  cache/weights/            Pretrained weights and base-training checkpoints
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
