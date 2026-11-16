# Research Info

Research plan for the EuroSAT few-shot benchmark. Where it differs from the [original proposal](docs/project-proposal.pdf), this file is the current version (see section 6).

**Status (2026-11-16):** the plan is agreed. The code and notebook still follow the previous plan. They are updated in step 2 (section 5).

## 1. Summary

We test how well four image models classify satellite land cover from 1, 5 or 10 labeled images per class.

Each model is used in three states, called baselines: untrained, ImageNet-pretrained, and pretrained then fine-tuned on satellite images. The model stays frozen during few-shot learning. Only a small classifier is trained on the labeled images.

We run three studies, called tracks, on every model and baseline: a model comparison, all augmentations together, and each augmentation alone.

## 2. Design at a glance

| | |
|---|---|
| Models (4) | ResNet-18, ConvNeXt-Tiny, ViT-B/16, DINO ViT-S/16 |
| Baselines (3) | Untrained · Pretrained · Fine-tuned |
| Tracks (3) | 1 Model comparison · 2 All augmentations · 3 Each augmentation |
| Shots | 1, 5 or 10 labeled images per class |
| Task | 4-way episodes, 15 query images per class |
| Episodes | 100 development + 600 test per shot |
| Where it runs | One local computer (RTX 5060 Ti, 16 GB). The notebook also keeps working on Colab |

Every track runs on all 4 × 3 = 12 backbones with the same episodes. Any two results can therefore be compared episode by episode.

## 3. Research questions

**Models (track 1)**
- Which model gives the best few-shot accuracy? Does the ranking change from 1 to 5 to 10 shots?
- Does ConvNeXt beat ResNet? Do the Transformers beat the CNNs? Does self-supervised DINO beat the supervised ViT?
- Evidence: T1, F2, F3. Class-level errors: F4, F5.

**Baselines**
- How much does ImageNet pretraining add over an untrained model?
- How much does fine-tuning on satellite images add over pretraining alone?
- Evidence: F11, T1.

**All augmentations (track 2)**
- Does training the classifier with augmented support images help? For which models and baselines?
- How much of the gap to more real labels does augmentation close?
- Evidence: F6, F7, F8.

**Each augmentation (track 3)**
- Which methods help, and which hurt? Which model–augmentation pairs work best?
- Does a method help some classes and hurt others? Do more views help?
- Evidence: F6, T3, F10, F9.

The IDs refer to [figure_n_table_sug.md](figure_n_table_sug.md).

## 4. Setup

### 4.1 Data

EuroSAT RGB: 27,000 Sentinel-2 tiles, 64×64 px, 10 classes. The split comes from the committed [splits_manifest.json](splits_manifest.json) and does not change.

| Use | Classes | Images |
|---|---|---|
| Few-shot evaluation | HerbaceousVegetation, Pasture, PermanentCrop, River | 3,000 development · 7,000 test |
| Fine-tuning | AnnualCrop, Forest, Highway, Industrial, Residential, SeaLake | 10,200 train · 3,400 val · 3,400 test |

- An episode has k support images and 15 query images per class. Supports and queries never overlap.
- Development and test images never mix. Images can recur across episodes.
- Each shot budget has its own episodes. The 1-shot images are not part of the 5-shot sets.
- The four evaluated classes never appear in fine-tuning.
- Input for every model: RGB, resized to 224×224 (bilinear), ImageNet normalization.

### 4.2 Models

| Model | Type | Pretraining | Embedding | Params |
|---|---|---|---:|---:|
| ResNet-18 | Standard CNN | ImageNet-1K, supervised | 512 | 11.2M |
| ConvNeXt-Tiny | Modern CNN | ImageNet-1K, supervised | 768 | 27.8M |
| ViT-B/16 | Transformer | ImageNet-1K, supervised | 768 | 85.8M |
| DINO ViT-S/16 | Transformer | ImageNet-1K, self-supervised (no labels) | 384 | about 21M |

- CNN embeddings are the pooled features before the classification head. Transformer embeddings are the final normalized class token.
- DINO ViT-S/16 replaces DINOv2, which was pretrained on a much larger dataset. All four models now share the same pretraining data.

### 4.3 Baselines

| Baseline | Backbone weights |
|---|---|
| Untrained | Random (seed 42) |
| Pretrained | ImageNet-1K |
| Fine-tuned | ImageNet-1K, then trained further on the six fine-tuning classes |

Each baseline is its own model instance. No model is shared between baselines, and no backbone ever trains on development or test images.

Fine-tuning recipe, the same for every model:
- Start from a fresh copy of the pretrained weights.
- Train the whole backbone plus a new 6-class head on the 10,200 train images, in the same order for every model.
- AdamW, cosine schedule, 10 epochs, batch 64, no augmentation.
- Try learning rates 1e-5, 3e-5 and 1e-4. Keep the run and epoch with the best accuracy on the 3,400 val images.
- Report accuracy on the 3,400 test images as a check. Then drop the 6-class head and freeze the backbone.

### 4.4 Few-shot classifier

- The backbone is frozen for all three baselines. Only the classifier is trained.
- Classifier: multinomial logistic regression with L2 regularization on L2-normalized embeddings.
- C is chosen from {0.01, 0.1, 1, 10, 100, 1000} by mean accuracy on the development episodes. It is chosen separately for each baseline, model, augmentation condition and shot. Ties go to the smaller C. A pick at 0.01 or 1000 is flagged.
- The chosen C is then fixed. A new classifier is fitted on each test episode's supports and scored on its queries.
- Test labels never choose any setting.

### 4.5 Tracks

| Track | Classifier training data, per support image | Control |
|---|---|---|
| 1 Model comparison | The original | — |
| 2 All augmentations | The original + one view from each of the 8 methods (9 rows) | The original repeated 9 times |
| 3 Each augmentation | The original + one view from one method (2 rows) | The original repeated twice |

- Track 2 is exactly the eight track-3 views combined.
- The control ("repeat") has the same number of rows but no new pixels. It separates the effect of new views from the effect of more rows.
- View sweep (extra run): 1, 2, 4 and 8 views per image for the six methods that can make unlimited distinct views. Flip stops at 2 and rotation at 3.

### 4.6 Augmentation

Rules:
- Applied to support images only. Queries are never augmented.
- Views do not count as shots.
- All models and baselines get the same views.
- No black edges, no duplicate views, and every view must differ clearly from its original.

| Method | What it does | Setting |
|---|---|---|
| flip | Mirror | left-right or top-bottom |
| rotation | Quarter turn | 90°, 180° or 270° |
| scale up | Zoom in on the center | ×1.1–1.3 |
| brightness/contrast | Change both | each ×0.7–0.9 or ×1.1–1.3 |
| noise | Gaussian pixel noise | std 0.05 |
| blur | Gaussian blur | σ 2.5–3.5 px |
| erasing | Gray square | side 20–30% of the image |
| cutout | Black square | side 20–30% of the image |

- Settings apply to the 224×224 input and are fixed before any run.
- Blur, noise, erasing and cutout strengths were checked by eye on real development support images.
- Translation, zoom-out and tilted rotation were dropped because they create black edges.

### 4.7 Metrics

- Accuracy and Macro-F1 on test queries, averaged over the 600 episodes.
- Per-class recall and confusion matrices.
- Paired differences in percentage points on the same episodes: model vs model, baseline vs baseline, augmentation vs none.
- 95% bootstrap confidence intervals over episodes.
- The share of episodes where an augmentation gives no gain.
- Model size and compute cost.

Two rules for reading results:
- Comparisons across shot budgets use means only, because each shot budget has its own episodes.
- The preferred augmentation for a baseline, model and shot is the one chosen on development episodes. The candidates are the 8 single methods and no augmentation. Its gain is then checked on test.

## 5. Research steps

| Step | What | Status |
|---|---|---|
| 1 | Agree the plan: this file and the figure plan | Figure plan is next |
| 2 | Update the code: three baselines per model, fine-tuning, the new augmentation set, the wider C search, tidy folders, a simple notebook | To do |
| 3 | Install GPU PyTorch on the local computer | To do |
| 4 | Smoke run: every stage end to end on a few episodes | To do |
| 5 | Fine-tune the four models (three learning rates each) | To do |
| 6 | Main run: tracks 1–3 on all 12 backbones | To do |
| 7 | View-sweep run | To do |
| 8 | Figures, tables and write-up | To do |

## 6. Changes from the original proposal

| Topic | Proposal | Now |
|---|---|---|
| Task | 10-way, 60/20/20 image split | 4-way episodes on 4 classes; the other 6 classes are for fine-tuning |
| Baselines | Frozen pretrained models; fine-tuning optional | Three baselines: untrained, pretrained, fine-tuned |
| Fourth model | DINOv2 ViT-S/14 | DINO ViT-S/16, so all models are ImageNet-1K-pretrained |
| Augmentation methods | Flip, rotation, translation, scaling, brightness/contrast, noise, blur, erasing, cutout | The same without translation; rotation in 90° steps; scaling is zoom-in only |
| Views | Original + 4 views | One view per method; a separate view sweep |
| Augmentation strength | Tuned on development data | Fixed before runs, after a visual check |
| Support nesting | 1-shot inside 5-shot inside 10-shot, where possible | Not nested; cross-shot comparisons use means |
| Classifier C | Not specified | 0.01–1000, chosen on development episodes |

## 7. Limitations

- The models differ in size (11M to 86M parameters). ViT-B/16 trained on ImageNet-1K alone is a weak ViT.
- The untrained baseline is one random draw per model.
- Fine-tuning uses different classes from the evaluated ones. It measures adaptation to satellite images, not knowledge of the test classes.
- Images recur across episodes, and the split is by image, not by location. Confidence intervals describe this episode set, not new regions.
- Only four classes are evaluated, and three of them are vegetation.
- The 64 px tiles are upsampled to 224 px.
- Many conditions are compared. We select on development data and report every condition, including negative results.

## 8. Where things live

| What | Where |
|---|---|
| Experiments | [EuroSAT_Representation_Benchmark.ipynb](EuroSAT_Representation_Benchmark.ipynb) |
| Settings and defaults | Notebook Section 1; [src/eurosat/config.py](src/eurosat/config.py) |
| Split | [splits_manifest.json](splits_manifest.json), [SPLITS.md](SPLITS.md) |
| Figure and table plan | [figure_n_table_sug.md](figure_n_table_sug.md) |
| Method details | [docs/](docs/) |

## 9. To resolve

| # | Item | Why it matters | Status |
|---|---|---|---|
| 1 | Team agreement to replace DINOv2 with DINO ViT-S/16 | It changes a model named in the proposal | Open |
| 2 | ResNet-18 code | The team has existing code to use. It is attached at step 2 | Waiting |
| 3 | DINO ViT-S/16 checkpoint | Pin the exact version and verify the embedding point, as done for the other models | Step 2 |
| 4 | Main-text figures and tables | Page limit; 14 items are planned | Open |
| 5 | Development-label reporting | The proposal promises it; the log currently records 0 | Step 2 |
