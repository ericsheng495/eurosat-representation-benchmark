# Part B: ConvNeXt-Tiny and geometric augmentation validation

This report documents the proposal-aligned Part B benchmark for the ConvNeXt-Tiny backbone. It follows the same structure and guardrails as the DINOv2 suite in [part-d.md](part-d.md): shared train/validation/test partitions, identical support budgets, fixed augmentation policies, and a clear separation between development-only selection and final held-out evaluation.

## Research question

Does a modern CNN backbone provide a stronger frozen representation than the ResNet-18 baseline on the EuroSAT few-shot benchmark, and how much of that gain is attributable to geometric support augmentation rather than the backbone itself?

The study uses a single frozen ConvNeXt-Tiny ImageNet-1K checkpoint, common L2-regularized multinomial logistic regression, and the exact same support/query partitioning as the rest of the benchmark. The backing comparison remains the matched ResNet-18 primary grid; no statement should attribute ConvNeXt gains purely to architecture without holding the data protocol and classifier rule fixed.

## Prespecified protocol

| Requirement | Implementation |
|---|---|
| Backbone | ConvNeXt-Tiny, ImageNet-1K pretrained checkpoint |
| Dataset | Ten EuroSAT classes, verified deduplication, deterministic 60/20/20 split |
| Shot budgets | 1, 5, and 10 original support images per class |
| Augmentation controls | none, repeated originals, flip, rotation, translation, scale |
| Shared downstream rule | L2 multinomial logistic regression with development-selected C |
| View policy | Support-only geometric views; query set remains deterministic and untouched |
| Evaluation | Accuracy and Macro-F1 on the held-out query partition |
| Comparisons | ConvNeXt versus ResNet for the same policies and budgets |

The serviceable full-grid size is:

- 10 conditions × 3 shot budgets × 20 paired trials = 600 classifier records for the standard primary grid
- 1-shot development selection remains separate from final query reporting
- smoke mode preserves the same conditions on a smaller sample with two trials and a limited per-class cap

## Recommended execution

The benchmark notebook should set the model assignment to the ConvNeXt backbone before execution. The same structure as the DINOv2 suite is preserved in the project helper code and runner interface:

```sh
python -m eurosat part-b --mode smoke
python -m eurosat part-b --mode full
```

In the notebook workflow, the matching assignment is:

```python
ASSIGNMENT = 'convnext_tiny'
EXPERIMENT = 'part-b'
```

This keeps the execution pattern consistent with the rest of the project while reusing the same dataset, cache, and reporting conventions.

## Calculated findings from the project instructions

These are the explicit calculation rules the Part B report should use once the full grid is run. They are based on the benchmark protocol, not on a fabricated result table.

### Primary paired effects

For each repeated support-query trial, compute the paired difference in the same metric as:

- accuracy gain (percentage points) = accuracy_augmented - accuracy_none
- macro_f1 gain (percentage points) = macro_f1_augmented - macro_f1_none

For model contrasts, compute:

- ConvNeXt minus ResNet = score_convnext - score_resnet

These differences must be paired on the same original supports, shots, and query examples so the trial-level comparison is valid.

### Augmentation interpretation

The main Part B contrasts are:

- flip versus no augmentation
- rotation versus no augmentation
- translation versus no augmentation
- scale versus no augmentation
- repeat control versus no augmentation
- original + 4 geometric views versus 5 originals at 1-shot

The key quality check is that view budgets are not silently turned into more labeled originals. The benchmark uses the same labels and support-image identities across conditions; only the transformed views change.

### Expected report structure

The final Part B export should contain:

- per-trial accuracy and Macro-F1 for all policies and shot budgets
- paired augmentation-gain tables for each model and support condition
- ConvNeXt-versus-ResNet paired contrasts
- class-level recall and confusion summaries
- stage timing, parameter count, and cache-state provenance
- development-selection notes and any nonpositive effect rows

## Interpretation limits

The benchmark remains subject to the same design constraints as the rest of the study:

- Trial standard deviation is not a confidence interval.
- Query images are reused across trials, so confusion counts are not independent samples.
- Development labels are additional selection data and must be reported separately from the nominal shot count.
- The split is image-level and does not establish geographic independence without external metadata.
- ConvNeXt and ResNet differences include architecture, model size, and pretraining history; they do not isolate representation family alone.

## Output expectation

The Part B workflow should export a suite directory under the project storage, analogous to the DINOv2 suite structure. The report should preserve:

- model metadata and checkpoint identity
- support/query split provenance
- policy strengths and selected view budgets
- trial-level summaries and paired gain tables
- figure exports for accuracy curves and augmentation effect plots

This repository does not ship a fully executed ConvNeXt full-data run in the local workspace, so the present document intentionally records the protocol, calculation rules, and expected outputs rather than fabricating final benchmark numbers.
