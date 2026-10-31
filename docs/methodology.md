# Experimental protocol

## What is implemented

- Four frozen backbones: ImageNet-1K ResNet-18, ConvNeXt-Tiny and ViT-B/16, plus official DINOv2 ViT-S/14. Backbone identifiers, the pinned DINOv2 source revision and software versions are saved with each experiment.
- All ten classes, exact decoded-RGB pixel duplicate checks before splitting, and deterministic stratified 60% support-pool / 20% development / 20% held-out test partitions. Conflicting duplicate labels stop preparation. Geographic independence is not established by this image-level split.
- Nested 1/5/10-shot support samples, shared across models and augmentation conditions. Full mode uses 20 paired trials. Augmented views do not count as new shots.
- Frozen backbones in evaluation mode. ResNet uses pre-classifier pooled features; ConvNeXt retains its checkpoint's final normalization and pooled representation; ViT and DINOv2 use the final normalized class-token representation. Features are L2-normalized.
- The same L2-regularized multinomial logistic-regression procedure for every model. C is chosen from `[0.1, 1, 10]` using development accuracy per trial; ties choose the smaller C. Development labels are additional selection data and are not included in the shot budget. Test labels do not select C or the preferred augmentation.
- Separate support-only augmentation conditions: flip, rotation, translation, scaling, brightness/contrast, noise, blur and erasing. Full mode adds four views per original support image. No-augmentation and repeated-original controls are included. Development and test images receive deterministic preprocessing only.
- Accuracy, Macro-F1, per-class recall, confusion matrices, and paired gains against each trial's no-augmentation baseline. JSON, CSV and development-selected augmentation tables are exported.

Smoke mode uses 40 real images per class, one trial, one augmented view, ResNet-18, and none/repeat/flip conditions. Its results only establish that the workflow executes. Full mode is a substantial grid: 4 models × 10 conditions × 3 shot budgets × 20 trials = 2,400 classifier results. Plan the grid before reporting research results.

## Storage and recovery

In Colab, persistent storage is a new folder:

```text
MyDrive/ColabNotebooks/cs7643/EuroSATRepresentationBenchmark/
  data/EuroSAT.zip            Downloaded once and checksum-verified
  cache/weights/             Downloaded backbone implementations and weights
  cache/dataset-*.json        Duplicate-check manifests
  cache/split-*.json          Exact image partitions
  cache/features/            Original/augmented embeddings, saved in chunks
  cache/experiments/         Trial manifests, numeric classifier checkpoints, completed results
  outputs/<experiment-id>/   results.json, summary.csv, development-selected-policies.json
```

Extracted images live under `/content/eurosat-representation-benchmark` for fast reads. They are recreated from the persistent archive when Colab discards its runtime disk. No duplicated class-folder splits are written to Drive.

Feature caches identify dataset contents, exact checkpoint IDs, preprocessing, augmentation policy, deterministic image/view seeds and software versions. Completed feature chunks survive interruption. Classifier parameters (`coef`, `intercept`, `classes`) are saved as numeric NPZ checkpoints; a JSON completion record is published afterward. A disconnected classifier trial repeats only that trial. Fully completed model results skip model loading and feature extraction on rerun.

Original and augmented features are cached separately. All models use the same generated augmented image for a given image/policy/view seed. Reruns may still verify the archive and extract raw images after a runtime reset, but completed inference and classifier work is reused. There is no automatic cache deletion. Increment `PIPELINE_VERSION` when changing pipeline semantics or augmentation strengths.

Each execution records hardware, device, software versions and UTC time alongside the experimental manifest. A cached result retains its original stage timings; a resumed execution is not a fresh accelerator benchmark.

Your Mac's cache is separate from Drive. The first Colab run does not automatically receive local data, weights or results.


## Experimental limits and extensions

Fixed initial augmentation strengths are: horizontal or vertical flip; rotation ±30°; translation ±22 pixels after resizing; scaling 0.8–1.2; brightness and contrast factors 0.7–1.3; Gaussian noise σ=0.05 before normalization; Gaussian blur with a 5×5 kernel and σ=0.1–1.5; and a square erasing region of side 22–67 pixels filled with 0.5 before normalization. These are explicit starting settings, not tuned optima.

Full mode implements the primary original-plus-four-view comparison. To perform the proposal's selected nine-view comparisons, prespecify the model/policy subset and rerun with `config.views = 9` and `config.shots = [1]`; the numeric configuration distinguishes these artifacts. Cutout and backbone fine-tuning are optional extensions and are not in the default grid. Hardware and wall-clock conditions affect timing; cold model-load/download time is included in `original_feature_seconds`. This measures an end-to-end extraction stage, not isolated accelerator throughput. Spatially related samples may remain across partitions because geographic grouping metadata has not been established.

See [validation](docs/validation.md) for executed checks and remaining cloud verification.

Existing Drive storage named `EuroSATLowData` is reused automatically to preserve earlier caches. New storage uses `EuroSATRepresentationBenchmark`; set `EUROSAT_STORAGE` to override it.
