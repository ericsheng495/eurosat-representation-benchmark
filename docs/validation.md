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

The earlier portable/reference copies were consolidated into `EuroSAT_Representation_Benchmark.ipynb`. The executor now saves outputs to that same notebook; the builder preserves matching executed outputs, or restores measured dataset previews when code changes. Superseded execution and backup notebook copies have been removed. Per-experiment `outputs/<experiment-id>/results.json` records the real trial results and configuration. `work/tests.log` and `work/environment.freeze.txt` retain unit-test and dependency information. Local evidence, large artifacts and environments are ignored by Git.

## Remaining cloud checks

No Google account was authenticated and no Colab GPU was allocated by this agent. You must connect a GPU runtime and authorize Drive in your own Colab/VS Code session. Actual NVIDIA CUDA execution and Drive mount behavior have not been tested from this local environment. The first notebook cell explicitly checks the remote NVIDIA GPU before importing the model stack; the subsequent device check executes a tensor operation.

The complete 2,400-result full-data research grid has not been executed. Smoke-run scores are execution checks and must not be reported as final study results. Optional nine-view comparisons, cutout, geographic grouping and fine-tuning are not part of the default validated grid. The methodology describes how selected nine-view runs can be configured and documents the remaining study limitations.

## Modular repository and exploration checks (2026-10-30)

- Generated self-contained notebook: every cell passed in a fresh local kernel in 8.0 seconds, including full-dataset audit, all six figures, Parts A–D and nine ResNet smoke results.
- Package-import research notebook: every cell passed in a fresh local kernel in 11.6 seconds. All four models produced 36 smoke records with actual pretrained adapters; matching original features were reused, missing augmented features were computed, and classifiers were fitted.
- Nine unit tests passed in 2.3 seconds. Added checks verify measured dataset counts, support-only image provenance, real PNG/PDF exports, overlap rejection, and refusal to merge mismatched, incomplete or duplicated teammate results. Existing interruption/recovery and helper-consistency checks remain.
- Full dataset exploration measured 27,000 native 64×64 RGB images, ten classes, no decoded-pixel duplicates or corrupt images, and no image/pixel overlap across 16,200 / 5,400 / 5,400 partitions. Six PNG/PDF figures, CSV counts and JSON provenance were exported; class and augmentation montages were visually inspected.
- Notebook regeneration check, dependency consistency check, and wheel/source-distribution build passed locally. CI configuration is included but has not run on GitHub.

Evidence: `work/unit-checks.log`, `work/build.log`, the unified notebook, and `docs/figures/`. Some stages reuse existing caches; these elapsed times are workflow checks, not accelerator throughput comparisons.

The unified `EuroSAT_Representation_Benchmark.ipynb` passed every cell in a fresh local kernel in 8.8 seconds after consolidation, including six live dataset figures and cached ResNet smoke results. Nine unit tests passed.

Readable-code revision: the notebook contains ten actual implementation cells, without encoded code or repository imports. A fresh local execution passed in 8.6 seconds; hardware/configuration, dataset tables, six figures, cached-run status and the nine-row ResNet smoke table are saved in the notebook. The original prototype only had saved connection/Drive messages; those historical connection states are not presented as current runtime outputs.

## Naming and cleanup verification

The project now lives in `Desktop/eurosat-representation-benchmark` and uses `EuroSAT_Representation_Benchmark.ipynb`. The local editable installation and entry points were repaired after moving the folder. Eleven tests, dependency consistency, notebook drift checks and wheel/source builds pass. A fresh local kernel completed the renamed notebook in 11.0 seconds, including all dataset figures and cached smoke results; that verification was stored separately as `work/rename-validation.json` so the main notebook retains the user's saved Colab T4 outputs. Existing Drive cache folders remain supported. The drift-check regression test confirms a failed check never overwrites a teammate's notebook edits.

## Part D implementation and rendered results

On 2026-10-30, the complete notebook passed in a fresh local MPS kernel in 39.3 seconds with the official pinned frozen DINOv2 ViT-S/14 adapter. The real-image smoke subset contained 400 EuroSAT tiles; ten policies × 1/5/10 shots × two paired trials produced 60 classifier result records with four added views. Existing original embeddings were reused; new augmented embeddings were computed. Four result figures and their tables are saved inline after the six dataset figures. PNG/PDF reports, CSV tables, development-selected policies and provenance were exported to the run's report folder.

Thirteen tests pass, including report arithmetic (paired gains), development-only policy selection, actual PNG/PDF output verification and incomplete-pair rejection. Notebook drift checks, dependency checks and package builds pass. These runs verify execution, not final DINOv2 research performance. Full-data/20-trial runs and four-model paper integration remain pending.

## Complete local Part D suite verification

The proposal-aligned suite passed in a fresh local MPS notebook kernel: 44 development-only strength scores, 60 main-grid classifier records, eight selected nine-view follow-up records, and 30 paired label/view-budget comparisons. The first complete execution took 52.6 seconds. A final cached rerun completed all cells in 15.8 seconds and reused the development selection plus both completed model grids without loading the backbone again. The saved notebook contains 13 figures (six dataset, six outcome, one budget-comparison figure) and no error outputs. Both primary and follow-up runs retain their complete exports and classifier checkpoints.

Nineteen tests pass. Added coverage verifies the full 600-record primary configuration, optional cutout support, development-only method/strength selection, selection resume, identical original supports across budgets, paired budget arithmetic, deterministic strength-dependent views, and cross-model differences in augmentation gains. Package builds, dependency checks and nonmutating notebook drift checks pass. Recorded costs include stage cache counts, parameter tensor size and practical memory scopes.

The full-data 20-trial suite is implemented but has not been executed. The completed evidence uses 40 tiles per class and two trials; it must not be reported as final study performance. Teammate A–C full-data exports, geographic metadata and final paper review remain external dependencies. These updates are local only; no Git push was performed.

## Expanded metrics and matched ResNet baseline

The classification evaluator now exports macro precision/recall, balanced accuracy, weighted F1, MCC/kappa and per-class precision/F1/query counts, alongside the proposal's accuracy/Macro-F1/recall/confusion metrics. Completed confusion matrices are upgraded without backbone loading or classifier refitting; a regression test also verifies numeric feature/classifier checkpoint modification times remain unchanged. Supplementary 95% percentile intervals resample support-trial identities with a minimum of five trials; smoke intervals remain unavailable.

Twenty-four tests pass, including agreement with scikit-learn on imbalanced ten-class data, undefined/missing-class behavior, deterministic intervals and matched ResNet/DINO suite configurations. The real-image smoke workflow produced 60 DINOv2 and 60 matched ResNet-18 primary results plus eight DINOv2 nine-view follow-ups. The final fully cached notebook run passed in 17.8 seconds. Section 8 displays the two-model comparison, expanded metric overview and per-class classification report; 14 figures are saved inline. Full-data 20-trial experiments remain unexecuted. Package/dependency/drift checks pass; no push was performed.
