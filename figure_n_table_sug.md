# Figure and Table Plan

The 15 figures and tables for the final report. The experiment design is in [research_info.md](research_info.md).

All 15 are built from a run's results by `src/eurosat/reporting/` (notebook Sections 6–8) and saved to `<storage>/outputs/<run>/report/`: figures as PNG and PDF, tables as CSV. An item the run cannot support (for example F3 without a model pair) is skipped with the reason. Example layouts: [docs/figure_examples.pdf](docs/figure_examples.pdf).

For each item:
- **What:** the kind of figure or table.
- **Contains:** what is plotted or listed.
- **Why:** the question it answers.
- **Where:** the report section and placement.

## Overview

| ID | Title | Kind | Report section | Placement | Data from |
|---|---|---|---|---|---|
| F1 | Augmented support views | Image grid | Methods | Supp. | No run needed |
| T2 | Models and fine-tuning | Table | Methods | Supp. | Fine-tuning + main run |
| T4 | Labels used | Table | Methods | Supp. | Main run |
| T1 | Main results | Table | Results: models | **Main** | Main run |
| F2 | Few-shot curves | Line chart | Results: models | **Main** | Main run |
| F3 | Paired model differences | Forest plot | Results: models | **Main** | Main run |
| F4 | Per-class recall | Heatmap | Results: models | Supp. | Main run |
| F5 | Confusion matrices | Matrix grid | Results: models | Supp. | Main run |
| F11 | Baseline effects | Forest plot | Results: baselines | **Main** | Main run |
| F6 | Augmentation gains | Heatmap | Results: augmentation | **Main** | Main run |
| F7 | All-augmentation gain by shot | Line chart | Results: augmentation | Supp. | Main run |
| F8 | Label-gap closure | Dot plot | Results: augmentation | **Main** | Main run |
| T3 | Preferred augmentation | Table | Results: augmentation | **Main** | Main run |
| F9 | Number of views | Line chart | Results: augmentation | Supp. | View-sweep run |
| F10 | Class-level augmentation effects | Heatmap | Results: augmentation | Supp. | Main run |

Main text: T1, F2, F3, F11, F6, F8, T3. The other eight are supplementary.

## Conventions

- Baselines are always named and ordered Untrained, Pretrained, Fine-tuned. A figure with baselines has one panel per baseline, in that order.
- Each model keeps one color in every figure.
- Uncertainty is a 95% confidence interval (bootstrap over episodes).
- "pp" means percentage points. A change from 70% to 75% is +5 pp.
- "Paired" means compared on the same episodes. Results from different shot budgets are compared by means only.

---

## Methods

### F1 · Augmented support views
- **What:** an image grid.
- **Contains:** one development support image per class (4 rows). Columns: the original, then one view from each of the 8 methods at its setting.
- **Why:** to show that every view stays recognizable, has no black edges, and differs clearly from its original.
- **Where:** Methods, augmentation. Supplementary.

### T2 · Models and fine-tuning
- **What:** a table, one row per model.
- **Contains:** type, pretraining data, parameters, embedding size. Compute cost of the pretrained backbone: embedding speed (images per second, loading and augmentation included), total classifier fitting time, and peak GPU memory. For fine-tuning: the chosen learning rate, 6-class val and test accuracy, and training time.
- **Why:** to give the context for reading model differences (size and cost), and to check that fine-tuning worked.
- **Where:** Methods, models and baselines. Supplementary.
- **Note:** times are wall-clock on one computer (RTX 5060 Ti, 8-core CPU). Classifier fits run in parallel on all CPU cores.

### T4 · Labels used
- **What:** a table, one row per shot budget, plus all shots and fine-tuning.
- **Contains:** labeled images by role. Train: the support images that fit each test classifier (the shot budget). Choose settings: the distinct development images that choose C and the preferred augmentation. Score only: the distinct test query images. The fine-tuning row lists its train, val and test images.
- **Why:** the proposal promises to report development labels separately from the shot budget. A 1-shot result uses 4 labels to train, but more labels to choose its settings.
- **Where:** Methods, evaluation protocol. Supplementary. Notebook Section 8.

## Results: models (track 1)

### T1 · Main results
- **What:** a table, one row per model and baseline (12 rows).
- **Contains:** accuracy at 1, 5 and 10 shots, as mean [95% CI], with Macro-F1 in the next column. No augmentation. Chance is 25%.
- **Why:** the headline numbers. It shows which model and baseline is best at each shot.
- **Where:** Results, first table. Main.
- **Note:** a high mean can hide a weak class. See F4.

### F2 · Few-shot curves
- **What:** a line chart, one panel per baseline.
- **Contains:** accuracy against shots (1, 5, 10). One line per model, a 95% CI band, and a chance line at 25%.
- **Why:** to show whether the model ranking changes with more labels, and whether it differs between baselines.
- **Where:** Results, models. Main.

### F3 · Paired model differences
- **What:** a forest plot (points with CI bars), one panel per baseline.
- **Contains:** three comparisons at each shot: ConvNeXt − ResNet, DINO − ViT-B/16, and Transformers − CNNs. Each point is the mean paired difference in pp.
- **Why:** the test behind "model A beats model B". These are the proposal's three comparisons.
- **Where:** Results, models. Main.
- **Note:** a bar that does not cross 0 means a reliable difference. Transformers − CNNs can be driven by one strong model, so read it with the other two rows.

### F4 · Per-class recall
- **What:** a heatmap, one panel per baseline.
- **Contains:** rows are models. Columns are the four classes at 1, 5 and 10 shots. Each cell is mean recall. No augmentation.
- **Why:** to show which classes are hard, and for which model.
- **Where:** Results, error analysis. Supplementary.

### F5 · Confusion matrices
- **What:** a grid of 4×4 matrices: one row per model, one column per baseline (12 matrices).
- **Contains:** true class against predicted class at 10 shots. Each cell is the share of the true class.
- **Why:** to show which classes get mixed up, such as HerbaceousVegetation and Pasture.
- **Where:** Results, error analysis. Supplementary.

## Results: baselines

### F11 · Baseline effects
- **What:** a forest plot with two panels.
- **Contains:** left panel: Pretrained − Untrained. Right panel: Fine-tuned − Pretrained. Rows are model × shot. Each point is the mean paired difference in pp with a 95% CI.
- **Why:** to measure what each training stage adds: ImageNet pretraining first, then fine-tuning on satellite images. This is the main result of the three-baseline design.
- **Where:** Results, baselines. Main.

## Results: augmentation (tracks 2 and 3)

### F6 · Augmentation gains
- **What:** a heatmap, one panel per baseline.
- **Contains:** rows: all augmentations together (track 2), each of the 8 methods (track 3), and the two repeat controls. Columns: model × shot. Each cell is the paired accuracy gain over no augmentation, in pp. Bold means the CI excludes 0.
- **Why:** one view of both augmentation tracks. It shows which methods help, for which model, baseline and shot.
- **Where:** Results, augmentation. Main.
- **Note:** the repeat rows should be near 0. With about 130 cells per panel, a few will be bold by chance, so look for consistent patterns.

### F7 · All-augmentation gain by shot
- **What:** a line chart, one panel per baseline.
- **Contains:** the track-2 gain over no augmentation, in pp, against shots. One line per model, with 95% CI bars and a zero line.
- **Why:** to show whether augmentation matters most when labels are scarcest.
- **Where:** Results, track 2. Supplementary.

### F8 · Label-gap closure
- **What:** a dot plot, one panel per baseline, one row per model.
- **Contains:** four accuracy points: 1-shot, 1-shot with all augmentations, 5-shot, and 10-shot. A label gives the share of the 1-shot to 5-shot gap that augmentation closes.
- **Why:** to express the value of augmentation in terms of real labels.
- **Where:** Results, track 2. Main.
- **Note:** shot budgets use different episodes, so these are means, not paired differences.

### T3 · Preferred augmentation
- **What:** a table, one row per baseline, model and shot (36 rows).
- **Contains:** the method chosen on development episodes, its development gain, its test gain as mean [95% CI], and the share of test episodes with no gain.
- **Why:** to answer which model–augmentation pair works best, without choosing on test data. It also shows when augmentation fails.
- **Where:** Results, track 3. Main.
- **Note:** candidates are the 8 single methods and "none". Track 2 and the repeat controls are not candidates.

### F9 · Number of views
- **What:** a line chart, one panel per baseline.
- **Contains:** gain over no augmentation at 1-shot, averaged over the four models, against views per support image (1, 2, 4, 8). One line per method, plus a gray repeat line with the same number of rows. Flip stops at 2 views and rotation at 3.
- **Why:** to show whether more views keep helping.
- **Where:** Results, track 3. Supplementary.
- **Note:** the gap between a method and the repeat line is the benefit of new views.

### F10 · Class-level augmentation effects
- **What:** a heatmap, one panel per baseline.
- **Contains:** rows: all augmentations together and each of the 8 methods. Columns: the four classes. Each cell is the recall change over no augmentation at 1-shot, in pp, averaged over the four models.
- **Why:** to show trade-offs that the average hides. A method can help one class and hurt another.
- **Where:** Results, augmentation. Supplementary.
- **Note:** these values are noisy. Treat single cells as hypotheses.
