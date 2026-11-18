# Research Info

The research plan for the EuroSAT few-shot benchmark. Where it differs from the [original proposal](docs/project-proposal.pdf), this file is the current version (section 6).

**Status (2026-11-18):** fine-tuning is done. The full run is being repeated with a wider C range; the first run's results were deleted. Next: the full run (step 6).

## 1. Summary

We test how well four image models classify satellite images when only 1, 5 or 10 labeled images per class are available.

- The models stay frozen. Only a small classifier learns from the labeled images.
- Each model is tested in three states, called **baselines**: untrained, ImageNet-pretrained, and fine-tuned on other satellite classes.
- Three studies, called **tracks**, run on every model and baseline: a model comparison, all augmentations together, and each augmentation alone.

Terms used below:

| Term | Meaning |
|---|---|
| Episode | One small 4-way task with its own labeled and unlabeled images |
| Support images | The labeled images in an episode. "k-shot" means k per class |
| Query images | The images the classifier must label: 15 per class |
| Backbone | One model in one baseline. There are 4 × 3 = 12 |
| Embedding | The backbone's output vector for one image |
| Development / test | Two separate image pools. Development chooses settings. Test only scores |

## 2. Design at a glance

| | |
|---|---|
| Models | ResNet-18, ConvNeXt-Tiny, ViT-B/16, DINO ViT-S/16 |
| Baselines | Untrained · Pretrained · Fine-tuned |
| Tracks | 1 Model comparison · 2 All augmentations · 3 Each augmentation |
| Shots | 1, 5 or 10 support images per class |
| Episodes | 100 development + 600 test per shot |
| Where it runs | One local computer (RTX 5060 Ti, 16 GB). The notebook also works on Colab |

All 12 backbones use the same episodes, so any two results can be compared episode by episode.

## 3. Research questions

The IDs refer to [figure_n_table_sug.md](figure_n_table_sug.md).

| Topic | Questions | Evidence |
|---|---|---|
| Models (track 1) | Which model is best, and does the ranking change with more shots? Does ConvNeXt beat ResNet? Do Transformers beat CNNs? Does self-supervised DINO beat the supervised ViT? | T1, F2, F3 (class errors: F4, F5) |
| Baselines | How much does ImageNet pretraining add? How much does fine-tuning on satellite images add? | T1, F11, and T5 for why |
| All augmentations (track 2) | Do augmented support images help, and for which models and baselines? How much of the gap to more real labels do they close? | F6, F7, F8 |
| Each augmentation (track 3) | Which methods help or hurt? Which model–method pairs work best? Do methods trade off between classes? Do more views help? | F6, T3, F10, F9 |

## 4. Setup

### 4.1 Data

EuroSAT RGB: 27,000 Sentinel-2 tiles, 64×64 px, 10 classes. The split is fixed in [splits_manifest.json](splits_manifest.json).

| Use | Classes | Images |
|---|---|---|
| Few-shot evaluation | HerbaceousVegetation, Pasture, PermanentCrop, River | 3,000 development · 7,000 test |
| Fine-tuning | AnnualCrop, Forest, Highway, Industrial, Residential, SeaLake | 10,200 train · 3,400 val · 3,400 test |

- In an episode, support and query images never overlap.
- Development and test images never mix. One image can appear in many episodes.
- Each shot budget has its own episodes. The 1-shot supports are not part of the 5-shot supports.
- The four evaluated classes are never used for fine-tuning.
- Every model gets RGB input resized to 224×224 (bilinear) with ImageNet normalization.

### 4.2 Models

| Model | Type | Pretraining (ImageNet-1K) | Embedding size | Parameters |
|---|---|---|---:|---:|
| ResNet-18 | Standard CNN | Supervised | 512 | 11.2M |
| ConvNeXt-Tiny | Modern CNN | Supervised | 768 | 27.8M |
| ViT-B/16 | Transformer | Supervised | 768 | 85.8M |
| DINO ViT-S/16 | Transformer | Self-supervised (no labels) | 384 | 21.7M |

- CNN embeddings are the pooled features before the classification head. Transformer embeddings are the final normalized class token.
- DINO ViT-S/16 replaces DINOv2, which was pretrained on far more data. Now all four models share the same pretraining data. Its checkpoint is pinned to commit 7c446df.

### 4.3 Baselines

| Baseline | Backbone weights |
|---|---|
| Untrained | Random (seed 42). No training |
| Pretrained | Official ImageNet-1K weights |
| Fine-tuned | ImageNet-1K, then trained on the six fine-tuning classes |

Each baseline builds its own model. No backbone ever trains on development or test images.

Fine-tuning recipe, the same for every model. It uses the stabilizers from the teammates' training code.
- Start from the pretrained weights. Train the whole backbone and a new 6-class head on the 10,200 train images. The data order is the same for every model.
- The new head starts at zero, so every model starts at the same loss. A random head would give DINO, whose embeddings are about 5× longer, much larger first gradients.
- AdamW with weight decay 0.05 (none on biases, normalization weights, position embeddings and class tokens). 10 epochs, batch 64, mixed precision.
- No augmentation, dropout or stochastic depth. ConvNeXt's default stochastic depth is turned off to match the other models.
- Learning rate: 0.1× in epoch 1 (warmup), then a cosine decay over epochs 2–10.
- Try 10 learning rates from 1e-6 to 1e-3. Keep the run and epoch with the best val accuracy. Test accuracy is a check only.
- Then drop the head and freeze the backbone.

### 4.4 Few-shot classifier

- Multinomial logistic regression with L2 regularization, on L2-normalized embeddings.
- C is chosen on the development episodes: the value with the best mean accuracy, ties to the smaller C. It is chosen separately for each baseline, model, augmentation condition and shot.
- The C candidates are ten values, one per decade from 0.001 to 1,000,000. A pick at either end is flagged.
- With C fixed, a new classifier is fitted on each test episode's supports and scored on its queries.
- Test labels never choose anything. The development labels used to choose settings are counted apart from the shot budget (T4).
- The first run used 0.01–1000 and picked 1000 in about half the groups. A check on development episodes showed that accuracy peaks near 1000 and falls well before the new ends, so the range was widened.

### 4.5 Tracks

| Track | Training rows per support image | Control ("repeat") |
|---|---|---|
| 1 Model comparison | The original | — |
| 2 All augmentations | The original + one view from each of the 8 methods (9 rows) | The original 9 times |
| 3 Each augmentation | The original + one view from one method (2 rows) | The original twice |

- Track 2 combines exactly the eight track-3 views.
- The repeat control has the same number of rows but no new pixels. It separates the effect of new views from the effect of more rows.
- View sweep, in the same run: 1, 2, 4 and 8 views per image for the six methods that can make many different views. Flip stops at 2 views and rotation at 3.

### 4.6 Augmentation

- Support images only. Queries are never augmented, and views do not count as shots.
- Every model and baseline gets the same views.
- No black edges, no duplicate views, and every view must differ clearly from its original.

| Method | What it does | Setting (on the 224×224 input) |
|---|---|---|
| flip | Mirror | left-right or top-bottom |
| rotation | Quarter turn | 90°, 180° or 270° |
| scale up | Zoom in on the center | ×1.1–1.3 |
| brightness/contrast | Change both | each ×0.7–0.9 or ×1.1–1.3 |
| noise | Gaussian pixel noise | std 0.05 |
| blur | Gaussian blur | σ 2.5–3.5 px |
| erasing | Gray square | side 20–30% of the image |
| cutout | Black square | side 20–30% of the image |

- Settings were fixed before any run. Blur, noise, erasing and cutout were checked by eye on development support images.
- Translation, zoom-out and tilted rotation were dropped because they create black edges.

### 4.7 Metrics

- Accuracy and Macro-F1 on the test queries, averaged over the 600 test episodes, with 95% bootstrap confidence intervals.
- Per-class recall and confusion matrices.
- Paired differences in percentage points (pp) on the same episodes: model vs model, baseline vs baseline, augmentation vs none. Also the share of episodes with no gain.
- Many-label check (T5): accuracy with all development labels, and how many directions the embeddings really use.
- Cost (T2): embedding speed, classifier fitting time and peak GPU memory. Labels used by role (T4).

Two rules for reading results:
- Different shot budgets use different episodes, so they are compared by means only.
- The preferred augmentation is chosen on development episodes from the 8 single methods and "none". Its gain is then checked on test.

## 5. Progress

| Step | What | Status |
|---|---|---|
| 1 | Agree the plan (this file and the figure plan) | Done. Two items are still open (section 8) |
| 2 | Update the code and the notebook | Done |
| 3 | Install GPU PyTorch locally | Done: conda env `eurosat`, PyTorch 2.15, CUDA 13.2 |
| 4 | Smoke run of every stage | Done |
| 5 | Fine-tune the four models | Done in 2.8 hours. Chosen learning rates: ResNet 2e-5, ConvNeXt 1e-5, ViT and DINO 5e-6, none at the edge. 6-class val 99.7–99.8% |
| 6 | Full run: tracks 1–3 and the view sweep on all 12 backbones | Rerun with the wider C range. The first run (1.6 hours) led to T5 and the new range |
| 7 | Figures, tables and write-up | To do |

## 6. Changes from the original proposal

| Topic | Proposal | Now |
|---|---|---|
| Task | 10-way, 60/20/20 image split | 4-way episodes on 4 classes; the other 6 classes are for fine-tuning |
| Baselines | Frozen pretrained models; fine-tuning optional | Untrained, pretrained and fine-tuned |
| Fourth model | DINOv2 ViT-S/14 | DINO ViT-S/16, so all models are ImageNet-1K-pretrained |
| Augmentation | Flip, rotation, translation, scaling, brightness/contrast, noise, blur, erasing, cutout | The same without translation; rotation in 90° steps; zoom-in only |
| Views | Original + 4 views | One view per method, plus a view sweep |
| Augmentation strength | Tuned on development data | Fixed before any run, after a visual check |
| Support nesting | 1-shot inside 5-shot inside 10-shot | Not nested; shot budgets are compared by means |
| Classifier C | Not specified | 0.001–1,000,000, chosen on development episodes |

## 7. Limitations

**Models and baselines**
- The models differ in size, from 11M to 86M parameters. ViT-B/16 trained on ImageNet-1K alone is a weak ViT.
- Every random choice uses one seed (42): one draw of untrained weights per model, one fine-tuning run per learning rate, and one view per image and method. Confidence intervals cover the spread over episodes only. Small gaps inside the untrained or fine-tuned baseline may not hold for another draw. Pretrained results depend only on the views.
- The untrained ConvNeXt is almost a shallow network. torchvision scales each block's output by 1e-6 at the start, so its blocks do almost nothing.
- Fine-tuning uses other classes than the evaluated ones. It measures adaptation to satellite images, not knowledge of the test classes.
- The fine-tuning learning rate is chosen by 6-class val accuracy. Most rates reach 99.6–99.8%, so this choice says little about how well the embeddings transfer.

**Data**
- Images recur across episodes, and the split is by image, not by place. Confidence intervals describe this episode set, not new regions.
- Only four classes are evaluated, and three of them are vegetation.
- The 64 px tiles are upsampled to 224 px.

**Method**
- Blur changes smooth tiles very little. In a check of 40 development supports, 12 blurred views (mostly Pasture) looked almost the same as the original. A stronger blur would erase class cues such as orchard texture.
- Many conditions are compared. Choices are made on development data, and every condition is reported, including negative results.

## 8. Open items

| # | Item | Why it matters |
|---|---|---|
| 1 | Team agreement to replace DINOv2 with DINO ViT-S/16 | It changes a model named in the proposal |
| 2 | Which figures and tables go in the main text | Page limit. The plan has 8 main items out of 16 |

## 9. Where things live

| What | Where |
|---|---|
| Experiments | [EuroSAT_Representation_Benchmark.ipynb](EuroSAT_Representation_Benchmark.ipynb) |
| Settings and defaults | Notebook Section 1; [src/eurosat/config.py](src/eurosat/config.py) |
| Split | [splits_manifest.json](splits_manifest.json), [SPLITS.md](SPLITS.md) |
| Figure and table plan | [figure_n_table_sug.md](figure_n_table_sug.md) |
| Fine-tuned weights | `cache/weights/finetuned/` |
| Results, figures and tables | `outputs/<run>/` (`results.json` and `report/`) |
