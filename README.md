# EuroSAT Low-Data Classification

A fresh CS7643 project based on [proposal draft 2](docs/project-proposal.pdf). This project uses frozen pretrained representations and all ten EuroSAT classes. It does not reuse the old fine-tuned six-base/four-novel-class notebook.

## Start here

1. Open **[Start_Here.ipynb](Start_Here.ipynb)** in Google Colab (upload this one file).
2. Choose **Runtime → Change runtime type → T4 GPU** or another available GPU. Connect.
3. Run all cells and authorize Google Drive. Leave `MODE = 'smoke'` for the first check.
4. After the check succeeds, change `MODE` to `'full'`, inspect the configuration, then Run all again.

In VS Code, open `EuroSAT.code-workspace`, open `Start_Here.ipynb`, then choose **Select Another Kernel → Colab → New Colab Server → GPU → Python 3 (ipykernel)**. Python is the kernel language; GPU is the server hardware. Choosing Python on a TPU server does not turn it into a GPU server. Google sign-in and runtime allocation happen in your account; this repository cannot allocate a server on your behalf.

The first cell checks for a remote NVIDIA GPU before importing PyTorch or downloading data. This prevents the earlier TPU/local-kernel confusion. The later hardware check performs a real tensor operation on the selected device. Connection details: [official Colab VS Code guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide).

## Project files

```text
Start_Here.ipynb             Self-contained cloud notebook; the only file you must upload
src/eurosat/pipeline.py      Canonical, readable implementation
scripts/build_notebook.py    Rebuild the notebook after editing pipeline helpers
scripts/run.py               Local smoke/full experiment runner
scripts/execute_notebook.py  Execute the actual notebook in a fresh local kernel
tests/test_pipeline.py       Leakage, cache recovery and resumability tests
docs/project-proposal.pdf    Your latest downloaded proposal
EuroSAT.code-workspace      VS Code workspace
```

The original project remains separately in `Desktop/eurosat-fewshot-benchmark`; it is not used by this folder.

## What is implemented

- Four frozen backbones: ImageNet-1K ResNet-18, ConvNeXt-Tiny and ViT-B/16, plus official DINOv2 ViT-S/14. Backbone identifiers, the pinned DINOv2 source revision and software versions are saved with each experiment.
- All ten classes, exact decoded-RGB pixel duplicate checks before splitting, and deterministic stratified 60% support-pool / 20% development / 20% held-out test partitions. Conflicting duplicate labels stop preparation. Geographic independence is not established by this image-level split.
- Nested 1/5/10-shot support samples, shared across models and augmentation conditions. Full mode uses 20 paired trials. Augmented views do not count as new shots.
- Frozen backbones in evaluation mode. ResNet uses pre-classifier pooled features; ConvNeXt retains its checkpoint's final normalization and pooled representation; ViT and DINOv2 use the final normalized class-token representation. Features are L2-normalized.
- The same L2-regularized multinomial logistic-regression procedure for every model. C is chosen from `[0.1, 1, 10]` using development accuracy per trial; ties choose the smaller C. Development labels are additional selection data and are not included in the shot budget. Test labels do not select C or the preferred augmentation.
- Separate support-only augmentation conditions: flip, rotation, translation, scaling, brightness/contrast, noise, blur and erasing. Full mode adds four views per original support image. No-augmentation and repeated-original controls are included. Development and test images receive deterministic preprocessing only.
- Accuracy, Macro-F1, per-class recall, confusion matrices, and paired gains against each trial's no-augmentation baseline. JSON, CSV and development-selected augmentation tables are exported.

Smoke mode uses 40 real images per class, one trial, one augmented view, ResNet-18, and none/repeat/flip conditions. Its results only establish that the workflow executes. Full mode is a substantial grid: 4 models × 10 conditions × 3 shot budgets × 20 trials = 2,400 classifier results. Plan the grid before reporting research results.

## Storage and recovery

In Colab, persistent storage is a new folder:

```text
MyDrive/ColabNotebooks/cs7643/EuroSATLowData/
  data/EuroSAT.zip            Downloaded once and checksum-verified
  cache/weights/             Downloaded backbone implementations and weights
  cache/dataset-*.json        Duplicate-check manifests
  cache/split-*.json          Exact image partitions
  cache/features/            Original/augmented embeddings, saved in chunks
  cache/experiments/         Trial manifests, numeric classifier checkpoints, completed results
  outputs/<experiment-id>/   results.json, summary.csv, development-selected-policies.json
```

Extracted images live under `/content/eurosat-low-data` for fast reads. They are recreated from the persistent archive when Colab discards its runtime disk. No duplicated class-folder splits are written to Drive.

Feature caches identify dataset contents, exact checkpoint IDs, preprocessing, augmentation policy, deterministic image/view seeds and software versions. Completed feature chunks survive interruption. Classifier parameters (`coef`, `intercept`, `classes`) are saved as numeric NPZ checkpoints; a JSON completion record is published afterward. A disconnected classifier trial repeats only that trial. Fully completed model results skip model loading and feature extraction on rerun.

Original and augmented features are cached separately. All models use the same generated augmented image for a given image/policy/view seed. Reruns may still verify the archive and extract raw images after a runtime reset, but completed inference and classifier work is reused. There is no automatic cache deletion. Increment `PIPELINE_VERSION` when changing pipeline semantics or augmentation strengths.

Each execution records hardware, device, software versions and UTC time alongside the experimental manifest. A cached result retains its original stage timings; a resumed execution is not a fresh accelerator benchmark.

Your Mac's cache is separate from Drive. The first Colab run does not automatically receive local data, weights or results.

## Local development

A fresh Python 3.11 environment is already installed at `.venv/` on this Mac. To recreate it elsewhere:

```sh
python3.11 -m venv .venv
./.venv/bin/python -m pip install -r requirements-local.txt
```

Local checks deliberately use CUDA, Apple MPS or CPU instead of a cloud connection:

```sh
./.venv/bin/python scripts/run.py
./.venv/bin/python scripts/execute_notebook.py
./.venv/bin/python -m unittest discover -s tests -v
```

To check all four models on the smoke dataset:

```sh
./.venv/bin/python scripts/execute_notebook.py --models resnet18,convnext_tiny,vit_b_16,dinov2_vits14
```

Rebuild `Start_Here.ipynb` after editing the canonical pipeline:

```sh
./.venv/bin/python scripts/build_notebook.py
```

The notebook includes the helpers for portability; a test verifies they match the module. Local execution output is saved in `work/Start_Here.executed.ipynb`. Large data, environments, caches and outputs are ignored by Git.

## Experimental limits and extensions

Fixed initial augmentation strengths are: horizontal or vertical flip; rotation ±30°; translation ±22 pixels after resizing; scaling 0.8–1.2; brightness and contrast factors 0.7–1.3; Gaussian noise σ=0.05 before normalization; Gaussian blur with a 5×5 kernel and σ=0.1–1.5; and a square erasing region of side 22–67 pixels filled with 0.5 before normalization. These are explicit starting settings, not tuned optima.

Full mode implements the primary original-plus-four-view comparison. To perform the proposal's selected nine-view comparisons, prespecify the model/policy subset and rerun with `config.views = 9` and `config.shots = [1]`; the numeric configuration distinguishes these artifacts. Cutout and backbone fine-tuning are optional extensions and are not in the default grid. Hardware and wall-clock conditions affect timing; cold model-load/download time is included in `original_feature_seconds`. This measures an end-to-end extraction stage, not isolated accelerator throughput. Spatially related samples may remain across partitions because geographic grouping metadata has not been established.

See [validation](docs/validation.md) for executed checks and remaining cloud verification.
