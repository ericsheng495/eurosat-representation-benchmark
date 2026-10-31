# Part D: complete DINOv2 experiment suite

The notebook contains the actual code. Choose `ASSIGNMENT = 'dinov2_vits14'`, `EXPERIMENT = 'part-d'`, and run in order. `MODE = 'full'` uses the full dataset and 20 paired trials; `MODE = 'smoke'` checks every stage on a small real-image subset. The suite resumes cached feature chunks, classifier trials and development-selection scores. No GitHub access is required to run it.

```sh
# Local validation of the entire workflow, including view-budget comparisons.
python -m eurosat part-d --mode smoke
# Full Part D on the local machine's CUDA/MPS/CPU device.
python -m eurosat part-d --mode full
# Optional proposal cutout ablation, kept separate in the experiment identity.
python -m eurosat part-d --mode full --include-cutout
```

In Colab, use the notebook's `TARGET = 'colab_gpu'`; the CLI intentionally uses local hardware. Change `MODE` to `full` after validation. Existing `EUROSAT_*` trial/policy overrides should be cleared when starting the standard full preset.

## What the suite runs

| Proposal requirement | Implementation |
|---|---|
| Official frozen DINOv2 ViT-S/14 | Pinned source revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, final normalized class token, L2-normalized embeddings, gradients disabled and evaluation mode |
| Ten classes and isolated partitions | Shared decoded-pixel duplicate audit and deterministic 60/20/20 support/development/query split |
| Shared 1/5/10 originals per class | Nested samples, identical originals and deterministic views across conditions/models |
| Conventional CNN baseline | ResNet-18 primary grid, using identical locked strengths, original supports, splits and classifier rules as DINOv2 |
| Common downstream classifier | L2 multinomial logistic regression; C chosen from the shared grid on development accuracy |
| Development-based augmentation strengths | At 1-shot, compare magnitude factors 0.5/1/1.5 using paired development scores; smaller strength breaks ties. Flip has no magnitude parameter. Query labels and query predictions are not used |
| Main augmentation study | Original only, repeated-original control, and eight individual policies; original plus four views for each augmented condition |
| 1 original + 4 views versus 5 originals | Paired trial comparisons for every primary augmentation/control |
| Selected 1 original + 9 views versus 10 originals | Two methods selected by primary 1-shot development accuracy; names break ties. Selection is recorded before the follow-up queries are scored |
| Increasing the number of views | Paired nine-versus-four-view comparisons on the same 1-shot originals, including repeated-original controls |
| Accuracy and Macro-F1 with variability | Means, trial SDs and paired gain SDs; all prespecified conditions remain visible |
| Class-specific benefits and harms | Per-class recall changes for every policy and shot budget, baseline confusion matrices and trial-level class-effect CSV |
| Computational cost | Parameter count, parameter-tensor bytes, separate model-load/feature-stage times, classifier fits, cache-hit counts and practical memory measurements |
| Negative results and provenance | Nonpositive effects, paired model contrasts when other models are merged, software/checkpoint/data manifests, result hashes, execution records and interpretation notes |
| Optional cutout | `INCLUDE_CUTOUT = True` or `--include-cutout`; never silently added to the primary grid |

The default full primary grid produces 10 conditions × 3 shot budgets × 20 trials = **600 classifier records**. Two selected nine-view methods plus baseline/repeated-original controls produce **80 follow-up records**. The matched ResNet-18 primary grid adds **600 baseline records** (disable explicitly with `--skip-resnet-baseline` only for a DINO-only report). Development tuning is a separate stage with **440 development-only fits** (C selection evaluates additional candidates within each fit procedure). Smoke uses two trials and 40 original tiles per class; it cannot support final scientific conclusions.

## Development labels and fair comparison

The 1-shot tuning supports use the same first originals as the primary trial's nested supports. The development partition supplies additional labeled selection data beyond the classifier's original-shot budget. Strengths are locked before the primary grid, and follow-up methods are selected without reading primary test accuracy. All models must use the same locked strengths for the eventual cross-model comparison: share `strength-selection.json` with Parts A–C and set their `config.policy_strengths` to its `selected_strengths`. This designated-backbone tuning procedure is an explicit operational choice; freeze it with the team before final runs.

Set `SELECT_STRENGTHS = False` or `--skip-strength-selection` only when the team has explicitly chosen the fixed initial strengths. It creates a different suite identity. Optional fine-tuning is a separate study and is not required for this frozen-backbone Part D.

## Outputs

Each suite saves `plan.json`, `strength-selection.json`, `nine-view-selection.json` and `suite.json` under `outputs/part-d/<suite-id>/`. Primary and follow-up classifier checkpoints and `results.json` exports remain in their individual experiment directories.

The report contains accuracy-by-shot, Macro-F1-by-shot, paired augmentation gains, class recall, confusion matrices and class-augmentation delta PNG/PDF figures. CSV exports include all trial scores, paired gains, class effects, nonpositive effects and cost stages. Development-selected policy tables, checkpoint/software/data provenance and an interpretation note accompany them.

`comparisons/budget-comparisons.csv` retains the trial pairs and distinct original-image/classifier-row budgets. `budget-summary.csv` reports means, population trial SDs and the fraction of negative differences; the corresponding figure shows label/view-budget comparisons. Row counts can match while the number of labeled originals differs—augmented views never become additional shots.

To combine teammate results and regenerate all shared plots:

```sh
python -m eurosat merge reports/resnet.json reports/convnext.json reports/vit.json reports/dino.json --output outputs/combined
python -m eurosat report outputs/combined/results.json
```

Merging requires compatible data, software, strengths, trials and all non-model settings. Never mix smoke and full exports, or four- and nine-view grids in that merger; the separate budget-comparison function verifies their paired 1-shot supports instead.

## Interpretation limits

Trial standard deviation is not a confidence interval. Queries are reused across trials, so summed confusion counts are not additional independent images. Class effects report both positive and negative changes. Development labels are extra labeled selection data. An image-level split does not establish geographic independence; geographic grouping remains unavailable without reliable metadata. DINOv2 versus ViT differences also include architecture, model size and pretraining data, so the comparison does not isolate self-supervision.

Extraction times distinguish cache reads and computation; do not sum a stage time repeated across classifier rows. Process RSS is a lifetime peak, CUDA allocated memory is a per-stage peak, and MPS allocated memory is an end-of-stage snapshot; report their different scopes. Execution metadata for older cached rows may not identify the original computation device.

Implementation completion is separate from full-data execution and paper review. The checklist records those statuses independently. All current work stays local; no push is performed.

Expanded metric definitions, per-class reports, model-baseline comparisons and conditional confidence intervals are documented in [metrics.md](metrics.md). The notebook displays the combined ResNet/DINO report by default.
