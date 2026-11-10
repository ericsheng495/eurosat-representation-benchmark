# Metrics and comparisons

All metrics are evaluated on held-out queries after classifier settings are chosen using development episodes. Labels follow the four novel classes in the committed manifest.

| Output | Meaning |
|---|---|
| Accuracy | Fraction of query images classified correctly |
| Macro-F1 | Equal-weight mean of four class F1 scores |
| Class precision, recall and F1 | Class-specific errors and recognition rates |
| Confusion matrix | True classes in rows, predicted classes in columns |
| Macro precision/recall, weighted F1, balanced accuracy | Supplementary class-balance diagnostics |
| MCC and Cohen's kappa | Supplementary agreement measures |
| Paired model difference | Same-episode accuracy difference against ResNet-18, in percentage points |
| Paired augmentation gain | Same-model/shot/episode augmented accuracy minus no-augmentation accuracy |
| Computational cost | Parameter counts/bytes, feature/classifier time and scoped memory when available |

JSON stores trial metrics and confusion counts. Reports export summary CSVs, class tables, paired comparisons and figures. The notebook also saves `resnet-baseline-comparison.csv`. Undefined per-class precision/recall/F1 use zero; undefined kappa is null. Fractions use [0, 1]; accuracy differences use percentage points.

## Baselines and optional augmentation

ResNet-18 is the CNN baseline. Every model uses the same manifest episodes, preprocessing, shot budgets and classifier selection rule. No augmentation is the default and the matched reference for any augmentation comparison. Repeated originals control for extra classifier rows without image variation. Strengths are fixed across models; no test result selects settings.

Budget comparisons are meaningful only when the corresponding original/view conditions have actually run. The default manifest experiment does not run an augmentation-strength search or a view-budget sweep. Reports must not interpret missing conditions as measured results.

## Variability

Across-episode standard deviations accompany means. Supplementary 95% percentile bootstrap intervals use 2,000 resamples with fixed seed 42; paired differences are formed within matching episodes before resampling. Intervals are unavailable below five episodes, so two-episode smoke runs have blank intervals. Five is an implementation threshold, not a coverage guarantee.

Full mode uses 600 test episodes per shot. Episodes can reuse images and supports need not be nested across shot budgets. Intervals describe variation across this fixed episode collection; they do not establish geographic generalization or account for all comparisons. Plot error bars are labeled as episode SD.

Saved confusion counts allow compatible older results to be enriched with classification metrics without reloading a backbone or refitting classifiers. Regenerate reports with `python -m eurosat report PATH/TO/results.json`. Cached feature timings retain their original execution context and should not be presented as new hardware measurements.
