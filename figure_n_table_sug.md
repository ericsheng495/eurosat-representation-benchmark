# Figure and Table Suggestions

Mock layouts only. No numbers here are results.

| ID | Shows | Answers | Placement | Status |
|---|---|---|---|---|
| T1 | Accuracy per model × shot | Transfer quality, best model | **Main** | Partly exists |
| F2 | Accuracy vs shots, one line per model | Ranking change 1→5→10 | **Main** | Exists |
| F3 | Paired model differences + CI | ConvNeXt vs ResNet, DINOv2 vs ViT, Transformers vs CNNs | **Main** | Partly exists |
| F6 | Augmentation gain heatmap (method × model × shot) | Which methods help, for which model, at which shot | **Main** | Partly exists |
| F8 | 1-shot vs 1-shot + 4 views vs 5-shot | How much of the label gap augmentation closes | **Main** | Needs extra runs |
| T3 | Dev-selected method, checked on test | Preferred method per model; negative results | **Main** | Partly exists |
| F1 | Example views per augmentation | Do augmentations keep the class recognizable? | Supp. | Partly exists |
| F4 | Recall heatmap (model × class) | Which classes are hard, for which model | Supp. | Partly exists |
| F5 | Confusion matrices (10-shot) | Which classes get confused | Supp. | Exists |
| T2 | Size, pretraining data, cost | Context for model differences | Supp. | Partly exists |
| F7 | Gain of the dev-selected method vs shots | Are gains larger at 1-shot? | Supp. | New |
| F9 | Accuracy vs number of views (0/1/4/9) | Do more views help? | Supp. | Needs extra runs |
| F10 | Recall change per class per method | Does a method help some classes, hurt others? | Supp. | Exists |

**Status:** Exists = `reporting.py` draws it · Partly exists = data saved, new format · New = computable from current runs · Needs extra runs = augmentation / `VIEWS` runs + merge code.

**Conventions**
- One fixed color per model in every figure.
- Uncertainty = 95% CI (bootstrap over episodes), not SD. "pp" = percentage points.
- Within a shot budget: paired (same episodes). Across shot budgets: means only.

---

## Main

**T1 · Main results** — accuracy per model × shot, no augmentation.
Columns: Model · Family · 1-shot · 5-shot · 10-shot (mean [95% CI], Macro-F1 below).
- *Mean accuracy:* overall ranking. Can hide weak classes (see F4).
- *95% CI:* precision of the mean. Overlapping CIs ≠ no difference (see F3).
- *Macro-F1:* each class weighted equally; catches a model that over-predicts one class.

**F2 · Few-shot curve** — accuracy vs 1/5/10 shots, one line per model.
- *Line crossing:* the ranking changes.
- *CI band:* uncertainty. Shots use different episodes, so compare means only.
- *25% line:* chance level for 4-way.

**F3 · Paired model differences** — mean difference on the same episodes, with CI.
- *Point right of 0:* the first model wins.
- *CI not crossing 0:* a reliable difference for these episodes (not proof for new regions).
- *Transformers − CNNs:* a family-level answer; one strong model can drive it, so read it with the single rows.

**F6 · Augmentation gain heatmap** — "with − without augmentation" (pp) per method × model × shot.
- *Blue / red:* gain / loss. *Bold:* CI excludes 0.
- *~120 cells:* a few will be bold by chance; trust consistent patterns.
- *repeat row:* the row-count control; should be ≈ 0.

**F8 · Label-gap closure** — ○ 1-shot, ◆ 1-shot + 4 views, ● 5-shot, per model.
- *Gap closed % = (◆ − ○) / (● − ○):* intuitive value of augmentation.
- Means over different episodes, not paired.
- Same layout for 1-shot + 9 views vs 10-shot.

**T3 · Dev-selected augmentation** — per model × shot.
Columns: Model · Shots · Dev-selected method · Dev gain · Test gain [CI] · Episodes with gain ≤ 0.
- *Dev selection:* chosen without test labels (no cherry-picking).
- *Test gain + CI:* does the choice hold on test?
- *Episodes with gain ≤ 0:* how often augmentation fails even when the average is positive.

## Supplementary

**F1 · Augmented support views** — one support image per method at default magnitudes.
- Catches harmful settings (black borders, invisible blur). One random draw, not the full range.

**F4 · Per-class recall** — model × class heatmap, one panel per shot.
- Rows show a model's weak spots; columns show generally hard classes. Ignores false positives (see F5).

**F5 · Confusion matrices** — ResNet-18 and the best model at 10 shots.
- *Diagonal* = recall; *off-diagonal* = which classes get mixed up. Summed over episodes.

**T2 · Model characteristics and cost**
Columns: Model · Pretraining data · Params · Embedding dim · Images/s · Peak GPU MB · Fit ms.
- Size and pretraining data explain, but do not cause, differences. Measure costs on the same GPU without cache hits.

**F7 · Gain of the dev-selected method by shot** — one line per model.
- *Downward slope:* augmentation matters most when labels are scarce. *0:* dev chose "none" (a valid negative result).

**F9 · Number of views** — accuracy at 0/1/4/9 views for selected methods + repeat.
- *Flattening:* more views stop helping (flip and 90° rotation plateau early: only 2 and 3 distinct views).
- *Gap above repeat:* the real diversity benefit.

**F10 · Class-level augmentation effects** — recall change per class × method (one model, one shot).
- *Mixed blue/red row:* a trade-off hidden by the average. Noisy: treat single cells as hypotheses.
