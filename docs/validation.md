# Validation

Checked on 2026-10-30 on the local Mac using Apple MPS, Python 3.11.5, PyTorch 2.8.0, Torchvision 0.23.0, NumPy 2.4.6 and scikit-learn 1.7.2. A fresh `.venv` was installed for this repository. The original project environment, data and model caches were not reused.

## Executed checks

1. Portable notebook, initial ResNet smoke run: passed every cell, including a fresh public EuroSAT download with checksum verification, duplicate checking over the complete dataset, deterministic subset/splits, real pretrained feature extraction, classifier fitting and saved output. 400 real images; 240 support-pool / 80 development / 80 held-out test. Nine condition/shot results. Elapsed: 47.8 seconds.
2. Four-model real-image smoke run: ResNet-18, ConvNeXt-Tiny, ViT-B/16 and pinned official DINOv2 ViT-S/14 all passed. None/repeat/flip × 1/5/10 shots × one trial produced 36 results. All four official pretrained weight sets downloaded and loaded successfully. Elapsed: 48.9 seconds.
3. ResNet all-policy check: all ten default conditions, three shot budgets, two paired trials and four added views completed on the real-image smoke split. Sixty result records, numeric classifier checkpoints, CSV/JSON metrics and development-selected policies were written. Elapsed: 16.9 seconds. This check includes the final budget-independent nested support-sampling implementation.
4. Fresh-kernel rerun of the same sixty-result check: passed all cells in 3.1 seconds. The pipeline reused completed results without loading a model or recomputing features/classifiers. All 145 existing numeric feature/checkpoint/pretrained-weight cache files retained their modification times.
5. Original seven unit tests passed: disjoint partitions and nested support sets (also across changed shot/view grids), deterministic augmentation inputs, exact-pixel duplicate removal before splitting, interrupted feature-chunk recovery, interrupted classifier-trial recovery and complete-model skip, atomic checkpoint preservation and portable notebook/helper consistency.
6. `pip check`: no broken requirements.

## Local evidence

`work/portable.executed.ipynb` and `work/reference.executed.ipynb` retain the latest executed portable and package-import notebooks. Per-experiment `outputs/<experiment-id>/results.json` records the real trial results and configuration. `work/tests.log` and `work/environment.freeze.txt` retain unit-test and dependency information. Local evidence, large artifacts and environments are ignored by Git.

## Remaining cloud checks

No Google account was authenticated and no Colab GPU was allocated by this agent. You must connect a GPU runtime and authorize Drive in your own Colab/VS Code session. Actual NVIDIA CUDA execution and Drive mount behavior have not been tested from this local environment. The first notebook cell explicitly checks the remote NVIDIA GPU before importing the model stack; the subsequent device check executes a tensor operation.

The complete 2,400-result full-data research grid has not been executed. Smoke-run scores are execution checks and must not be reported as final study results. Optional nine-view comparisons, cutout, geographic grouping and fine-tuning are not part of the default validated grid. The methodology describes how selected nine-view runs can be configured and documents the remaining study limitations.

## Modular repository and exploration checks (2026-10-30)

- Generated self-contained notebook: every cell passed in a fresh local kernel in 8.0 seconds, including full-dataset audit, all six figures, Parts A–D and nine ResNet smoke results.
- Package-import research notebook: every cell passed in a fresh local kernel in 11.6 seconds. All four models produced 36 smoke records with actual pretrained adapters; matching original features were reused, missing augmented features were computed, and classifiers were fitted.
- Nine unit tests passed in 2.3 seconds. Added checks verify measured dataset counts, support-only image provenance, real PNG/PDF exports, overlap rejection, and refusal to merge mismatched, incomplete or duplicated teammate results. Existing interruption/recovery and helper-consistency checks remain.
- Full dataset exploration measured 27,000 native 64×64 RGB images, ten classes, no decoded-pixel duplicates or corrupt images, and no image/pixel overlap across 16,200 / 5,400 / 5,400 partitions. Six PNG/PDF figures, CSV counts and JSON provenance were exported; class and augmentation montages were visually inspected.
- Notebook regeneration check, dependency consistency check, and wheel/source-distribution build passed locally. CI configuration is included but has not run on GitHub.

Evidence: `work/unit-checks.log`, `work/build.log`, the two executed notebooks, and `docs/figures/`. Some stages reuse existing caches; these elapsed times are workflow checks, not accelerator throughput comparisons.
