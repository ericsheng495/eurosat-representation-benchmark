# Research Info

The current research plan for the EuroSAT few-shot benchmark. It refines the [original proposal](docs/project-proposal.pdf) with the decisions made since, and lists the steps from here to the final report.

## 1. Summary

We compare four frozen, pretrained image backbones (two CNNs and two Vision Transformers) for satellite land-cover classification when only 1, 5 or 10 labeled images per class are available. A common logistic-regression classifier is trained on each backbone's embeddings. We then test whether augmenting the few labeled support images helps, and whether the benefit depends on the model and the number of labels.

## 2. Research questions

**Part A: Representations (no augmentation)**

| # | Question | Main evidence |
|---|---|---|
| A1 | How well do ImageNet-pretrained representations transfer to satellite images? | T1, F2 |
| A2 | Does a modern CNN (ConvNeXt-T) beat a standard CNN (ResNet-18)? | F3 |
| A3 | Do the Transformers beat the CNNs under label scarcity? | F3 |
| A4 | Does DINOv2 (self-supervised) add value over a supervised ViT? | F3, T2 |
| A5 | Does the ranking change from 1 to 5 to 10 shots? | F2 |
| A6 | Which classes are hard, and for which models? | F4, F5 |

**Part B: Support-image augmentation**

| # | Question | Main evidence |
|---|---|---|
| B1 | Which augmentation methods beat no augmentation? | F6, T3 |
| B2 | Do models benefit from different methods? | F6, T3 |
| B3 | Are gains larger at 1-shot than at 5 or 10 shots? | F7, F6 |
| B4 | How much of the gap to more real labels does augmentation close? | F8 |
| B5 | Do more augmented views help further? | F9 |
| B6 | Does a method help some classes and hurt others? | F10 |

Figure and table details: [figure_n_table_sug.md](figure_n_table_sug.md).

**How to compare:** comparisons within one shot budget (model vs model, method vs none) are **paired** on identical episodes. Comparisons across shot budgets (A5, B3, B4) compare **means only**, because each shot budget has its own episodes.

## 3. Setup

### 3.1 Data and episodes

EuroSAT RGB: 27,000 Sentinel-2 tiles, 64×64 px, 10 classes. The archive is checksum-verified; the pixel audit found no exact duplicates and no corrupt images.

All splits come from the committed [splits_manifest.json](splits_manifest.json) (seed 42). Nobody regenerates it.

| Item | Value |
|---|---|
| Evaluated classes (4) | HerbaceousVegetation, Pasture, PermanentCrop, River |
| Reserved classes (6) | AnnualCrop, Forest, Highway, Industrial, Residential, SeaLake (not used) |
| Development pool | 3,000 images (30% of the 4 classes) |
| Test pool | 7,000 images (70%), disjoint from development |
| Episode | 4-way; k support + 15 query images per class |
| Shot budgets | k = 1, 5, 10 |
| Episodes per shot | 100 development, 600 test |

- Supports and queries never overlap within an episode.
- Images can recur across episodes. For example, the 600 test episodes at 1-shot use 2,004 distinct support images, and the queries cover 6,949 of the 7,000 test images.
- Supports are **not nested** across shot budgets: the 1-shot images are not part of the 5-shot set.

### 3.2 Preprocessing

The same input pipeline for every model: RGB → bilinear resize to 224×224 → ImageNet mean/std normalization.

### 3.3 Models

All backbones are frozen in evaluation mode with their classification heads removed. The embedding points below were verified to match the original models exactly.

| Model | Role | Checkpoint | Embedding | Dim | Params |
|---|---|---|---|---:|---:|
| ResNet-18 | Standard CNN (baseline) | torchvision `IMAGENET1K_V1` | global-average-pooled features | 512 | 11.2M |
| ConvNeXt-Tiny | Modern CNN | torchvision `IMAGENET1K_V1` | pooled features after final LayerNorm | 768 | 27.8M |
| ViT-B/16 | Supervised Transformer | torchvision `IMAGENET1K_V1` | final normalized class token | 768 | 85.8M |
| DINOv2 ViT-S/14 | Self-supervised Transformer | `facebookresearch/dinov2` (pinned revision) | official normalized class token | 384 | 22.1M |

### 3.4 Classifier and model selection

- **Classifier:** multinomial logistic regression with L2 regularization (`lbfgs`) on L2-normalized embeddings. It is identical for every model.
- **Choosing C:** C is picked from {0.1, 1, 10} by mean accuracy over the 100 development episodes, separately for each model × method × shot. Ties go to the smaller C.
- **Testing:** the chosen C is locked. A classifier is fitted on each test episode's supports and scored on its queries. Test labels never choose any setting.
- **Development labels** are extra information beyond the shot budget. We report this openly.

### 3.5 Augmentation

The rules follow the proposal:
- Augmentation is applied to **support images only**. Development and test queries are never augmented.
- Augmented views **do not count as shots**.
- Each original support image gets `VIEWS` augmented views (default 4, the proposal's main condition).
- Every model sees **the same augmented views**: each view's random seed depends on the image, the method and the view number.
- **`none`** is the baseline. **`repeat`** copies each original `VIEWS + 1` times: the same number of classifier rows, but no new pixels. It controls for the row-count effect.

| Method | What it does (on the 224×224 image) | Default magnitude |
|---|---|---|
| flip | Mirror left-right or top-bottom | — |
| rotation | Turn by 90°, 180° or 270° (lossless) | — |
| translation | Shift x and y independently | up to ±10% (±22 px) |
| scale | Zoom in or out | ×0.8 – 1.2 |
| brightness_contrast | Change brightness and contrast | each ×0.7 – 1.3 |
| noise | Add Gaussian pixel noise | std 0.05 |
| blur | Gaussian blur, kernel 5 | σ 0.1 – 1.5 px |
| erasing | Gray square, side 10–30% of the image | fill 0.5 |
| cutout | Black square, side 10–30% of the image | fill 0.0 |

Magnitudes are fixed **before** any run. They are not tuned on test or development scores. Defaults live in `AUGMENTATION_DEFAULTS` ([config.py](src/eurosat/config.py)); the notebook's Settings cell (Section 1) shows and overrides them, and every result records the values used.

### 3.6 Metrics and statistics

- **Main metrics:** accuracy and Macro-F1 on test queries, averaged over the 600 episodes.
- **Per class:** precision, recall, F1 and confusion matrices.
- **Effects:** paired difference in percentage points (pp), such as model A − model B or method − none, on the same episodes.
- **Uncertainty:** 95% percentile bootstrap over episodes (2,000 resamples).
- **Reliability:** the share of episodes where an augmentation gives no gain (≤ 0).
- **Cost:** parameters, extraction throughput, classifier fit time and peak memory.

## 4. Research steps

| Step | What | Settings (notebook) | Produces | Status |
|---|---|---|---|---|
| 0 | Lock the setup | Manifest, adapters, preprocessing, magnitudes | — | **Done** |
| 1 | Smoke run | `MODE='smoke'`, `POLICIES` = none + one method | End-to-end check | To do |
| 2 | Baseline run | `MODE='full'`, `POLICIES=['none']` | T1, F2–F5, T2 | To do |
| 3 | Augmentation run | `MODE='full'`, all methods + `repeat`, `VIEWS=4` | F6, F7, F8, F10, T3 | To do |
| 4 | View sweep | `VIEWS=1` and `9` for selected methods + `repeat` | F9, F8 (1-shot + 9 views vs 10-shot) | Needs merge code |
| 5 | Resolve remaining items (section 9) | Rerun only what the fixes require | Final numbers | After runs |
| 6 | Figures, tables, write-up | — | Report | After runs |

Notes:
- Step 3 includes `none`, so it can replace step 2. Running step 2 first gives early results while the costlier augmentation features are computed.
- Embeddings are cached on the shared Drive. Use one writer per experiment at a time.
- Choose the methods for step 4 from step 3's **development** results, not test results.

## 5. Changes from the original proposal

| Topic | Proposal | Now | Why / effect |
|---|---|---|---|
| Task | 10-way, stratified 60/20/20 image split | 4-way episodes on 4 classes; 6 classes reserved | The team shares one fixed manifest. Results cover 4 (mostly vegetation) classes, not all 10. |
| Trials | Not specified | 600 test and 100 development episodes per shot | Stable means and CIs |
| Support nesting | 1-shot ⊂ 5-shot ⊂ 10-shot "where possible" | Not nested; queries also differ by shot | Cross-shot comparisons (A5, B3, B4) are means, not paired |
| Development data | 20% split for selecting settings | 100 development episodes per shot; select C and preferred method | Same episode format as test |
| Rotation | Not specified | 90° steps only | Overhead images have no "up"; no black corners |
| Augmentation strength | Tuned on development data | Fixed before runs; visible in the notebook's Settings cell | Simpler, no strength search |
| Cutout | Optional | Implemented; differs from erasing only by fill color | See R5 in section 9 |

## 6. Open issues

All open issues and pending decisions are tracked in one place: [section 9](#9-to-resolve).

## 7. Limitations

- **Pretraining confounds:** the models differ in pretraining data, objective and size. ViT-B/16 vs DINOv2 does not isolate the effect of self-supervision. Our ViT-B/16 checkpoint was trained on ImageNet-1K only, which is weak for a ViT.
- **Image reuse:** images recur across episodes, and the split is by image, not by location. CIs describe this episode set, not new regions.
- **Scope:** 4 classes, 3 of them vegetation. Results may not hold for urban or water classes.
- **Resolution:** 64 px tiles are upsampled to 224 px for every model.
- **Many comparisons:** about 120 method × model × shot cells. We select on development data and report every condition, including negative ones.

## 8. Where things live

| What | Where |
|---|---|
| Run the experiments | [EuroSAT_Representation_Benchmark.ipynb](EuroSAT_Representation_Benchmark.ipynb) (built by [scripts/build_notebook.py](scripts/build_notebook.py)) |
| Run settings and augmentation magnitudes | Notebook Section 1 (Settings); defaults in [src/eurosat/config.py](src/eurosat/config.py) |
| Split definition | [splits_manifest.json](splits_manifest.json), [SPLITS.md](SPLITS.md) |
| Model adapters | [src/eurosat/models/](src/eurosat/models/) |
| Figure and table plan | [figure_n_table_sug.md](figure_n_table_sug.md) |
| Methods detail | [docs/methodology.md](docs/methodology.md), [docs/metrics.md](docs/metrics.md) |

## 9. To resolve

Items to settle as a team. Update **Status** (Open → Decided: …) as we agree, and move decided items to the list below.

**Decided so far**
- Split: the committed 4-way manifest; nobody regenerates it.
- Models and checkpoints as in 3.3; embedding points verified.
- Rotation: 90° steps only.
- Augmentation magnitudes are fixed before runs; no automatic tuning on development data.

### 9.1 Before the main runs (changes what we compute)

| # | To resolve | Why it matters | Options (suggested first) | Status |
|---|---|---|---|---|
| R1 | Range of C for logistic regression | {0.1, 1, 10} may cut off some model's best C, which would make the model comparison unfair | **Widen to 0.01–1000 (log steps) and flag picks at the edge**; keep as is | Open |
| R2 | Repeated views for flip and rotation | flip has 2 and rotation 3 distinct outputs, so `VIEWS=4` must repeat views. These methods then look weaker by construction | **Use distinct transforms per image**; merge them into one 8-way method (all flips × 90° turns); keep and report | Open |
| R3 | Black borders from translation and zoom-out | Queries never have black borders, so the classifier may learn an artifact | **Reflect padding**; crop-based shift; keep | Open |
| R4 | Blur strength | σ ≤ 1.5 px on a 3.5× upsampled image is about 0.4 original pixels, so the view is nearly a copy | **Raise σ after a visual check (F1)**; drop blur | Open |
| R5 | Erasing vs cutout | They differ only in fill color, so comparing them adds little | **Drop cutout** (optional in the proposal); make erasing use random size, shape and fill | Open |
| R6 | Other magnitudes (brightness/contrast, noise, scale, translation) | Defaults were set without data. Too weak gives no effect; too strong destroys the class | **Visual check on development supports, then lock**; small development sweep | Open |
| R7 | Run ownership and schedule | One writer per experiment on the shared Drive; GPU time is limited | Assign an owner to each step in section 4 | Open |

### 9.2 Before the view sweep (step 4)

| # | To resolve | Why it matters | Options (suggested first) | Status |
|---|---|---|---|---|
| R8 | Code to merge runs with different `VIEWS` | F8 (1-shot + 9 views vs 10-shot) and F9 need several runs side by side | **Write a merge-and-report step** | Open |
| R9 | Which methods enter the sweep | Running every method at `VIEWS` 1 and 9 is costly | **Top 2 development-selected methods per model + repeat** | Open |

### 9.3 Reporting and interpretation

| # | To resolve | Why it matters | Options (suggested first) | Status |
|---|---|---|---|---|
| R10 | Development-label accounting | The proposal promises to report development-label use; the log currently says 0 | **Report the number of distinct development images used (up to 3,000)** | Open |
| R11 | How to show uncertainty | Current plots show SD; CIs are standard. The bootstrap ignores image reuse across episodes | **95% CI everywhere + a reuse caveat** | Open |
| R12 | Unused reserved classes | 17,000 images (63%) are never used; reviewers may ask why only 4 classes | **State as a scope limitation**; extra evaluation on those classes | Open |
| R13 | ViT-B/16 checkpoint | `IMAGENET1K_V1` is a weak ViT and may understate Transformers | **Keep (matches the proposal) and state the limitation**; add a stronger ViT as an extra | Open |
| R14 | Fine-tuning extension | Optional in the proposal; costly | **Skip unless time remains** | Open |
| R15 | Main-text figures and tables | Page limit | **T1, F2, F3, F6, F8, T3** ([figure_n_table_sug.md](figure_n_table_sug.md)) | Open |
