# Experimental methods

## Source and fixed episodes

The persistent EuroSAT RGB archive is verified against its published MD5. Extracted files are checked against archive sizes and CRCs; missing or changed files are restored locally. Images are decoded to RGB for exact-pixel duplicate detection; conflicting labels and corrupt images stop execution.

The committed manifest defines six reserved base classes and four evaluated novel classes. Base train/validation/test partitions are retained for provenance but never train these frozen backbones. Novel development and test pools are disjoint. Each fixed four-way episode contains 1, 5 or 10 support originals per class and 15 query images per class. Supports and queries cannot overlap within an episode. Images can recur across episodes, and supports need not be nested across shot budgets. See [SPLITS.md](../SPLITS.md).

All pools and all episodes are validated even during smoke execution. Selected image records and episode assignments are exported under `cache/data-audits/`, alongside manifest/archive fingerprints and check results. Smoke selects the first two development/test episodes per shot; full uses 100 development and 600 test episodes per shot. This is a four-class task, distinct from the ten-class image split described in the original proposal.

## Representations and classifier

All models use the exact same episodes and input pipeline: RGB, bilinear resize to 224×224, and ImageNet channel normalization. ResNet-18 returns pooled 512-dimensional features; ConvNeXt-Tiny returns 768-dimensional features after its classifier normalization and flattening. ViT-B/16 returns the final normalized 768-dimensional class token. DINOv2 ViT-S/14 returns the official normalized 384-dimensional class token. Classification heads are removed. All backbone parameters remain frozen and models stay in evaluation mode. Embeddings are checked for shape/finite values and L2-normalized.

Multinomial L2-regularized logistic regression is common to every model. C is selected from `[0.1, 1, 10]` using mean accuracy across development episodes; ties choose the smaller C. For test evaluation, the selected C is locked, and a classifier is fitted separately to each test episode's supports before predicting its queries. Development labels are extra selection data and are not included in the stated shot budget.

The default policy is no augmentation. Optional fixed-strength policies affect supports only; development/test queries use deterministic preprocessing. Original images and augmented views keep the same partition. Generated views are shared across model conditions; augmented views never count as additional shots. Repeated-original controls match classifier row counts. No test score selects a strength or policy.

## Reporting and limitations

The notebook shows each model independently, then pairs episode scores against ResNet-18. Reports include accuracy, Macro-F1, class metrics, confusion matrices and measured computational costs. Uncertainty describes variation across the fixed manifest episodes; reused images prevent interpreting episodes as independent geographic samples. See [metrics](metrics.md).

Checkpoints, software versions, image/view identities and experiment settings are recorded with results. Compatible cached features and completed trials are reused; cached timings are not fresh execution benchmarks. Atomic writes do not provide a distributed lock for simultaneous Google Drive writers.

Image-level separation does not establish geographic independence. Model comparisons also reflect differences in pretraining data and model size. Smoke scores demonstrate execution only.
