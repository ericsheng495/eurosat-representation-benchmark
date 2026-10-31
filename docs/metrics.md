# Metrics and proposal baselines

Accuracy, Macro-F1, per-class recall, confusion matrices, paired augmentation gains, label/view-budget comparisons and computational costs are required by the [project proposal](project-proposal.pdf). Precision, weighted F1, balanced accuracy, agreement measures and confidence intervals are supplementary diagnostics. All are evaluated on held-out queries after settings are chosen on development data.

| Metric | Meaning | Stored output |
|---|---|---|
| Accuracy | Fraction of query images correctly classified | Trial JSON, summary CSV and baseline curves |
| Macro precision | Average of the ten class precision values, with equal class weight | Summary table and metric overview |
| Macro recall | Average recall across the ten classes | Summary table and metric overview |
| Macro-F1 | Average of ten class F1 scores | Summary table and curves |
| Weighted F1 | Class F1 scores weighted by their true query counts | Summary table and metric overview |
| Balanced accuracy | Average recall over classes represented in the queries; equals macro recall when all ten classes are present | Summary table and metric overview |
| Per-class precision | Among predictions of a class, fraction that are correct | `classification-report.csv` |
| Per-class recall | Among true images of a class, fraction correctly recognized | Class table, curves and effect heatmaps |
| Per-class F1 | Harmonic mean of that class's precision and recall | `classification-report.csv` |
| Class support | Number of true query images per class **per trial**, not the classifier's original-shot budget | `query_images_per_trial` in class report |
| MCC | Multiclass Matthews correlation coefficient | Summary CSV, with across-trial SD |
| Cohen's kappa | Observed agreement adjusted for agreement expected from label/prediction marginals | Summary CSV; undefined cases are null |
| Confusion matrix | True class (rows) versus predicted class (columns) | JSON counts and normalized heatmap |
| Paired augmentation gain | Same-trial augmented accuracy minus unaugmented accuracy, in percentage points | Paired trials, summary, SD and interval columns |
| Model-baseline difference | DINOv2 versus ResNet accuracy and augmentation-gain differences using matched trial identities | `model-comparisons.csv` and `model-comparison-summary.csv` |
| Label/view-budget difference | Paired 1+4 versus 5 originals, 1+9 versus 10 originals, and 9 versus 4 views | Budget comparison CSVs and figure |
| Compute cost | Parameters, parameter bytes, feature/classifier time, cache hits, device and scoped memory measurements | Cost and provenance exports |

Definitions follow [scikit-learn's classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics). Counts use all ten fixed class labels; undefined per-class precision/recall/F1 are zero. Undefined kappa is null rather than an invented score. Precision, recall, F1 and accuracy are fractions in [0, 1]; MCC and kappa can be negative. Accuracy differences use percentage points.

## Uncertainty

Every condition reports across-trial standard deviation. Supplementary 95% percentile bootstrap intervals resample support-trial identities with replacement (2,000 replicates, fixed seed 42). Paired differences are formed within a trial before bootstrapping, preserving the common supports and query set. The procedure follows the [percentile bootstrap definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); it uses NumPy and adds no dependency.

Intervals are exported for accuracy, Macro-F1, paired augmentation gains, model contrasts and budget contrasts. They are unavailable below five trials; two-trial smoke scores have blank intervals. Five is an explicit implementation threshold, not a guarantee of reliable coverage. Final full mode uses 20 trials. These intervals describe support-sampling variability **conditional on the fixed query images**. They do not establish spatial/generalization uncertainty, adjust for all model/policy comparisons, or turn smoke results into research evidence. Plot error bars remain trial SD and are labeled accordingly.

## Comparisons specified by the proposal

- **ResNet-18** is the conventional CNN representation baseline. The Part D suite now runs its primary grid alongside DINOv2, with identical data, nested original supports, strengths, seeds and classifier rules. ResNet receives no independently optimized test settings. The report shows both models and exports paired contrasts.
- **No augmentation** is each model/shot/trial's reference. Every augmentation gain uses this matched row.
- **Repeated originals** control for increasing classifier rows without adding image variation, matching the four-/nine-view counts.
- **5 and 10 original images per class** provide the label-budget references for the 1+4 and selected 1+9 conditions. Augmented views are never counted as new labeled originals.

Strengths are selected on development labels using the designated DINOv2 backbone and then shared unchanged. This tuning convention must be disclosed and agreed by the team. ViT-B/16 and ConvNeXt results are integrated when teammates provide compatible exports; no unavailable results are fabricated.

The standard suite includes the ResNet baseline (`INCLUDE_RESNET_BASELINE = True`). If a compatible ResNet export already exists locally, the same run is reused from cache. Disable it explicitly only when generating a DINO-only report; the suite identity records that choice.

## Reusing cached results

All added classification metrics can be calculated exactly from the saved confusion counts. Existing completed trials are enriched without loading a backbone or refitting their classifier. Confidence intervals use the saved trial scores. Numeric feature/classifier checkpoints remain intact. Metrics absent from a legacy export without confusion counts remain unavailable.

A normal suite rerun refreshes the JSON/CSV exports and notebook figures. `python -m eurosat report PATH/TO/results.json` can also regenerate report tables from an older completed export.
