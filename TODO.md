# Proposal-aligned work plan

Source: [project proposal](docs/project-proposal.pdf). Choose actual owners/reviewers in [collaboration.md](docs/collaboration.md); notebook Parts A–D link to implementation files. Checked items mean implemented, not that final research experiments are complete.

## Shared foundations

- [x] Four frozen pretrained backbones and one common logistic-regression procedure.
- [x] All ten classes, verified archive, decoded-pixel deduplication and deterministic 60/20/20 pools.
- [x] Nested paired 1/5/10-shot supports and support-only deterministic augmentation.
- [x] No-augmentation and repeated-original controls; original-plus-four-view full preset.
- [x] Cached weights, feature chunks, classifier checkpoints and completed trial recovery.
- [x] Full-dataset audit tables, single image, ten-class montage, cleaning/shuffling and augmentation figures.
- [x] Accuracy, Macro-F1, class recall, confusion matrices, paired gains and compatible export merging.
- [ ] Assign four owners and reviewers; confirm GitHub target and access.
- [ ] Freeze shared manifests, software, view counts, strengths and development-selection rules.

## Part A — Data and ResNet-18

- [ ] Independently review split isolation and reproduce dataset audit.
- [ ] Establish geographic grouping if metadata permits, or document image-level split limitations.
- [ ] Run full ResNet grid: all ten conditions, three shot budgets, 20 trials.
- [ ] Export complete results and checkpoint/software provenance; obtain reviewer signoff.

## Part B — ConvNeXt and geometric policies

- [ ] Review flip/rotation/translation/scale strengths using development data only.
- [ ] Run the same complete grid on ConvNeXt-Tiny, including all non-geometric conditions.
- [ ] Compare paired ResNet/ConvNeXt gains and check feature/cache consistency.
- [ ] Review Part A's shared data protocol.

## Part C — ViT and pixel/color policies

- [ ] Verify frozen ViT preprocessing, class-token extraction and feature dimensions.
- [ ] Review brightness/contrast, noise, blur and erasing strengths on development data.
- [ ] Run the same full grid on ViT-B/16 and export paired CNN/Transformer comparisons.
- [ ] Prespecify selected 1-shot nine-view comparisons against 5/10 originals; run without changing original support samples.
- [ ] Decide whether optional cutout merits a separate ablation.

## Part D — DINOv2 and report integration

- [ ] Run the same full grid on DINOv2 ViT-S/14; verify pinned checkpoint identity.
- [ ] Merge four complete compatible exports and verify paired trial completeness.
- [ ] Produce model-by-shot, augmentation-gain and class-level figures with variability.
- [ ] Report elapsed time, hardware, software and cold/warm-cache conditions.
- [ ] Explain extra development labels and pretraining-data/model-size confounds.
- [ ] Include negative results, geographic limitations, methods and dataset/checkpoint citations.
- [ ] Review final paper and reproduce reported tables from saved exports.

## Optional separate extensions

- [ ] Add fine-tuning only under a separately declared protocol and budget.
- [ ] Add ridge/Euclidean/cosine heads only if the team agrees an additional ablation is useful.

Do not add the original six-base/four-novel split or three-epoch training to the default frozen-backbone benchmark. See [migration coverage](docs/migration.md).
