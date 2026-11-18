# Figures and Tables

The 16 figures and tables for the final report. The experiment itself is described in [research_info.md](research_info.md).

- All 16 are made automatically from one full run and the fine-tuning records. They are saved to `outputs/<run>/report/`: figures as PNG and PDF, tables as CSV.
- 8 items go in the main text. The other 8 are supplementary.
- The notebook shows main items by default and supplementary items with `SHOW_DETAILS = True`. T4 is always shown in Section 8.
- If a run cannot support an item, the item is skipped and the reason is given.

## Overview

| ID | Title | Kind | Placement | Question it answers |
|---|---|---|---|---|
| **Methods** | | | | |
| F1 | Augmented support views | Image grid | Supp. | What do the augmented images look like? |
| T2 | Models and fine-tuning | Table | Supp. | How big and costly is each model, and did fine-tuning work? |
| T4 | Labels used | Table | Supp. | How many labels does each role use? |
| **Results: models and baselines** (notebook Section 6) | | | | |
| T1 | Main results | Table | **Main** | Which model and baseline is most accurate at each shot? |
| T5 | Many-label check | Table | **Main** | Why does fine-tuned score below pretrained? |
| F2 | Few-shot curves | Line chart | **Main** | Does the model ranking change with more labels? |
| F3 | Paired model differences | Forest plot | **Main** | Is one model reliably better than another? |
| F4 | Per-class recall | Heatmap | Supp. | Which classes are hard, and for which model? |
| F5 | Confusion matrices | Matrix grid | Supp. | Which classes get mixed up? |
| F11 | Baseline effects | Forest plot | **Main** | What do pretraining and fine-tuning each add? |
| **Results: augmentation** (notebook Section 7) | | | | |
| F6 | Augmentation gains | Heatmap | **Main** | Which methods help, for which model, baseline and shot? |
| F7 | All-augmentation gain by shot | Line chart | Supp. | Does augmentation matter most with the fewest labels? |
| F8 | Label-gap closure | Dot plot | **Main** | How many real labels is augmentation worth? |
| T3 | Preferred augmentation | Table | **Main** | Which method is best for each model, chosen without test data? |
| F9 | Number of views | Line chart | Supp. | Do more views keep helping? |
| F10 | Class-level augmentation effects | Heatmap | Supp. | Does a method help one class and hurt another? |

## How to read them

- Baselines always appear in the order Untrained, Pretrained, Fine-tuned. A figure with baselines has one panel per baseline.
- Each model keeps the same color in every figure.
- Error bars and brackets are 95% confidence intervals (bootstrap over test episodes).
- "pp" means percentage points: 70% → 75% is +5 pp.
- "Paired" means compared on the same episodes. A paired difference whose interval does not cross 0 is reliable.
- Different shot budgets use different episodes, so they are compared by means only.
- Chance accuracy is 25% (4 classes).

---

## Methods

### F1 · Augmented support views
- **Shows:** one development support image per class (rows). Columns: the original, then one view from each of the 8 methods.
- **Purpose:** to show that every view stays recognizable, has no black edges, and differs clearly from its original.

### T2 · Models and fine-tuning
- **Shows:** one row per model. Type, pretraining, parameters and embedding size. Cost: embedding speed (images per second), classifier fitting time and peak GPU memory. Fine-tuning: chosen learning rate, 6-class val and test accuracy, training time.
- **Purpose:** context for model differences, and a check that fine-tuning worked.
- **Read it:** times are wall-clock on one computer (RTX 5060 Ti, 8-core CPU).

### T4 · Labels used
- **Shows:** one row per shot budget, plus all shots and fine-tuning. Three columns of labeled images: those that train each classifier (the shot budget), those that only choose settings (development), and those that only score (test).
- **Purpose:** the proposal promises to report development labels apart from the shot budget. A 1-shot classifier trains on 4 labels, but its settings are chosen with many more.

## Results: models and baselines

### T1 · Main results
- **Shows:** one row per baseline and model (12 rows). Accuracy at 1, 5 and 10 shots as mean [95% CI], each with Macro-F1. No augmentation.
- **Purpose:** the headline numbers.
- **Read it:** a high mean can hide a weak class (see F4).

### T5 · Many-label check
- **Shows:** the same 12 rows. Next to 10-shot accuracy from T1, three checks on the embeddings, with no augmentation:
  - accuracy when the classifier trains on all development images (about 750 per class) and is scored on all test images. C is chosen by 5-fold cross-validation inside the development images.
  - effective dimensions: how many directions the embeddings really use.
  - the share of variance in the top 5 directions. A six-class head needs exactly five.
- **Purpose:** T1 shows fine-tuned well below pretrained. T5 shows whether fine-tuning lost the information or only made it hard to find.
- **Read it:** read the accuracy and the dimensions together.
  - In the first run, fine-tuned embeddings used about 5 directions, against 20–36 for pretrained. Yet with many labels, fine-tuned was about as accurate as pretrained. The information is still there, but 1–10 labels cannot find it.
  - Untrained embeddings also used very few directions, but stayed weak with many labels. Their information is missing.
  - A separate check confirmed that the fine-tuned weights load correctly.

### F2 · Few-shot curves
- **Shows:** accuracy against shots (1, 5, 10), one panel per baseline. One line per model with a 95% CI band, and a chance line.
- **Purpose:** to see whether the model ranking changes with more labels, and whether it differs between baselines.

### F3 · Paired model differences
- **Shows:** one panel per baseline. At each shot: ConvNeXt − ResNet, DINO − ViT-B/16, and Transformers − CNNs, as mean paired differences in pp.
- **Purpose:** the test behind "model A beats model B". These are the proposal's three comparisons.
- **Read it:** one strong model can drive Transformers − CNNs, so read it with the other two rows.

### F4 · Per-class recall
- **Shows:** one panel per baseline. Rows: models. Columns: the four classes at 1, 5 and 10 shots. Cells: mean recall. No augmentation.
- **Purpose:** to find the hard classes, and the models that struggle with them.

### F5 · Confusion matrices
- **Shows:** a 4×4 matrix for each model and baseline (12), at 10 shots. Each row is one true class, split by the label it received (shares).
- **Purpose:** to see which classes get mixed up, such as HerbaceousVegetation and Pasture.

### F11 · Baseline effects
- **Shows:** two panels. Left: Pretrained − Untrained. Right: Fine-tuned − Pretrained. Rows: model × shot. Mean paired differences in pp.
- **Purpose:** to measure what each training stage adds. This is the main result of the three-baseline design.
- **Read it:** the two panels have different scales. T5 explains the right panel.

## Results: augmentation

### F6 · Augmentation gains
- **Shows:** one panel per baseline. Rows: all augmentations (track 2), each of the 8 methods (track 3), and the two repeat controls. Columns: model × shot. Cells: paired accuracy gain over no augmentation, in pp. Bold means the interval excludes 0.
- **Purpose:** one view of both augmentation tracks.
- **Read it:** the repeat rows should be near 0. With about 130 cells per panel, a few are bold by chance, so look for consistent patterns.

### F7 · All-augmentation gain by shot
- **Shows:** the track-2 gain over no augmentation against shots, one panel per baseline, one line per model.
- **Purpose:** to see whether augmentation matters most when labels are scarcest.

### F8 · Label-gap closure
- **Shows:** one panel per baseline, one row per model. Four accuracy points: 1-shot, 1-shot with all augmentations, 5-shot and 10-shot. A label gives the share of the 1-shot to 5-shot gap that augmentation closes.
- **Purpose:** to express the value of augmentation in real labels.
- **Read it:** these are means over different episodes, not paired differences.

### T3 · Preferred augmentation
- **Shows:** one row per baseline, model and shot (36 rows). The method chosen on development episodes, its development gain, its test gain as mean [95% CI], and the share of test episodes with no gain.
- **Purpose:** to name the best method for each model without choosing on test data, and to show when augmentation fails.
- **Read it:** the candidates are the 8 single methods and "none".

### F9 · Number of views
- **Shows:** one panel per baseline. The 1-shot gain over no augmentation, averaged over the models, against 1, 2, 4 and 8 views per support image. One line per method, plus a gray repeat line with the same number of rows.
- **Purpose:** to see whether more views keep helping.
- **Read it:** the gap between a method and the repeat line is the value of new pixels. Flip stops at 2 views and rotation at 3.

### F10 · Class-level augmentation effects
- **Shows:** one panel per baseline. Rows: all augmentations and each of the 8 methods. Columns: the four classes. Cells: the 1-shot recall change over no augmentation, in pp, averaged over the models.
- **Purpose:** to show trade-offs that the average hides.
- **Read it:** single cells are noisy. Treat them as hypotheses.
