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
- [x] Create the private GitHub repository and configure the remote.
- [ ] Assign four owners and reviewers; grant teammate access.
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

## Part D — DINOv2 and complete proposal experiments

- [x] Official pinned frozen DINOv2, normalized class tokens and common logistic regression.
- [x] Complete main grid: ten conditions, nested 1/5/10 shots, four added views and paired trials.
- [x] Development-only strength selection, score caching and a locked shared configuration.
- [x] Development-only selection of follow-up methods before nine-view test evaluation.
- [x] Compare 1+4 views with 5 originals, 1+9 views with 10 originals, and 9 versus 4 views.
- [x] Accuracy/Macro-F1 variability, paired gains, class-specific augmentation effects and confusion matrices.
- [x] Add macro/per-class precision, F1 and query support, balanced accuracy, weighted F1, MCC and kappa.
- [x] Add conditional 95% support-trial bootstrap intervals with a minimum-trial threshold.
- [x] Include the matched ResNet-18 primary-grid baseline with paired model contrasts.
- [x] Record parameter/tensor size, model loading, extraction/fitting time, cache hits and practical memory use.
- [x] Export negative findings, paired model contrasts, software/checkpoint provenance and interpretation notes.
- [x] Support the optional cutout comparison without changing the default primary grid.
- [x] Resumable suite runner and readable notebook execution/visualization cells.
- [x] Meaningful tests for development selection, matched original supports and budget arithmetic.
- [x] Execute and visualize the entire smoke suite; verify a fully cached rerun.
- [x] Complete full-data Part D on Colab T4: 600 DINOv2 + 600 ResNet-18 primary records and 80 nine-view follow-ups.
- [ ] Independently review the completed full-data report.
- [ ] Have the team freeze/share strengths and agree on development-selection rules.
- [ ] Merge all four compatible full-data exports after Parts A–C complete.
- [ ] Independently review geographic/pretraining limitations and reproduce the final paper tables.

## Optional separate extensions

- [x] Add a separate fine-tuning notebook, matched neural-head control and epoch-resumable checkpoints.
- [x] Select DINOv2 final-block adaptation and map the extension to proposal pages 3–5.
- [x] Implement matched frozen/adapted × none/repeat/flip comparisons and the optional two-block ablation.
- [x] Run all 18 DINOv2 smoke fits and save comparison, learning-curve and class-error figures.
- [ ] Lock development-selected optimizer settings, run the 360-fit full extension and review it separately.
- [ ] Report exploratory scope, extra development labels and negative or inconsistent fine-tuning gains.
- [ ] Add ridge/Euclidean/cosine heads only if the team agrees an additional ablation is useful.

Do not add the original six-base/four-novel split or three-epoch training to the default frozen-backbone benchmark. See [migration coverage](docs/migration.md).
