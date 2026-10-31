# Original notebook coverage

The pasted Colab prototype is the starting reference. Its useful capabilities are represented once in the proposal-based implementation. The original prototype remains available separately; its six-base/four-novel experiment is a different protocol and should not be mixed into the primary results.

| Original section | Current location | Decision |
|---|---|---|
| Drive setup, ZIP reuse and extraction | Notebook section 2; `cache.prepare_storage`, `data.prepare_data` | Retained and improved: mount Drive, create storage, automatic public checksum-verified download, cached archive and local extraction. No Kaggle credentials, recursive Drive search or conflicting hard-coded project roots. |
| Six base / four novel classes | `data.stratified_split` | Replaced by all ten classes and proposal's 60/20/20 pools. No duplicated class folders. Base/novel protocol is an optional separate study, not an additional default split. |
| Seeded 70/15/15 training split and counts | `data.py`, `exploration.py` | Replaced by one persisted stratified split and nested paired support samples. Class counts, native images, duplicates and overlap checks appear before experiments. |
| Data loaders and ImageNet normalization | `features.Images`, `features.extract`, `augment.image_tensor` | Retained: batched tensors, deterministic query preprocessing and support-only augmentation. No random query transformation. |
| ResNet initialization and three-epoch fine-tuning | `models.load_backbone` | Pretrained ResNet retained; backbone training replaced by frozen adapters, as required by the proposal. Add fine-tuning only as a separately specified extension. |
| Saved training weights / fresh-session recovery | `models.py`, `cache.py`, `pipeline.py` | Explicit pretrained checkpoint identities plus cached features and numeric classifier checkpoints. The prototype's evaluation instantiated `weights=None` and left `load_state_dict` commented out, so its evaluation did not use the trained ResNet. That bug is not reproduced. |
| 1/5/10-shot episodes | `data.make_trials` | Retained with nested original supports, paired seeds across models/policies and fixed held-out queries. Full mode uses 20 trials rather than five independently sampled episodes per method. |
| Ridge, Euclidean and cosine centroids | `evaluation.fit_classifier` | Replaced by the proposal's common development-tuned multinomial logistic regression. These alternative heads remain optional ablations; they are not duplicate default classifiers. |
| Mean and standard-deviation accuracy | `evaluation.score`, `evaluation.summarize` | Retained and extended with Macro-F1, class recall, confusion matrices, paired augmentation gains and JSON/CSV exports. |
| Training/evaluation timings | Runtime and stage records in `pipeline.py` | Retained for extraction/classification; no training epochs exist in the frozen-backbone protocol. Report cold/cache conditions when comparing costs. |

The primary implementation adds ConvNeXt, ViT and DINOv2, matched augmentation controls, chunk-level recovery, dataset figures and teammate report merging. Full research experiments and optional extensions remain checkboxes in TODO.md; existing code is not evidence that those experiments have been completed.
