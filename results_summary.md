# Results Summary

Final full run, 2026-11-20. Figures and tables: `outputs/374799…/report/`. Design: [research_info.md](research_info.md). Figure and table IDs: [figure_n_table_sug.md](figure_n_table_sug.md).

Setup: 4 frozen models × 3 baselines (untrained, ImageNet-pretrained, fine-tuned on the 6 other EuroSAT classes). 4-way episodes on the 4 held-out classes, at 1, 5 and 10 shots, with 600 test episodes per shot. Chance is 25%.

## Main findings

1. **ImageNet pretraining is the biggest effect:** +18 to +38 pp over untrained.
2. **Fine-tuning on the six other classes hurts:** −9 to −23 pp below pretrained, for every model and shot. The information is not lost. It is squeezed into about five directions that 1–10 labels cannot use (T5).
3. **DINO is the best pretrained model** at 5 and 10 shots. It beats the supervised ViT-B/16 by 3.5–6.8 pp while being 4× smaller. ConvNeXt does not beat ResNet, and Transformers do not beat CNNs on average.
4. **Augmentation helps pretrained models only a little** (+0.6 to +1.9 pp with all methods). It helps fine-tuned models much more (+4.4 to +8.6 pp).
5. **The best method depends on the baseline:** 90° rotation for pretrained, blur and scale up for fine-tuned. Noise and blur slightly hurt pretrained models.

## Accuracy (T1)

Test accuracy (%) with no augmentation. Each cell: 1-shot · 5-shot · 10-shot. The 95% CIs are about ±0.3 to ±0.9 pp.

| Model | Untrained | Pretrained | Fine-tuned |
|---|---|---|---|
| ResNet-18 | 40.8 · 53.4 · 58.6 | 64.6 · 84.3 · 88.3 | 41.4 · 61.6 · 70.7 |
| ConvNeXt-Tiny | 37.4 · 48.2 · 50.8 | 60.0 · 83.0 · 88.3 | 40.9 · 61.8 · 71.7 |
| ViT-B/16 | 39.4 · 54.9 · 63.0 | 57.6 · 80.5 · 86.6 | 43.6 · 68.3 · 77.1 |
| DINO ViT-S/16 | 40.2 · 56.3 · 63.0 | 64.4 · 86.0 · 90.1 | 45.8 · 67.8 · 77.0 |

## Baselines (F11, T5)

- Pretrained − Untrained ranges from +18.2 to +37.5 pp. The largest gap is ConvNeXt at 10 shots, partly because the untrained ConvNeXt is almost a shallow network.
- Fine-tuned − Pretrained ranges from −9.4 to −23.2 pp. ViT-B/16 loses the least (−9 to −14). ResNet-18 loses the most (−18 to −23).

T5 shows why fine-tuning hurts:

| Baseline | 10-shot accuracy | Accuracy with all 2,999 development labels | Effective dimensions |
|---|---|---|---|
| Untrained | 51–63% | 57–84% | 1.5–2.2 |
| Pretrained | 87–90% | 95–97% | 20–36 |
| Fine-tuned | 71–77% | 94–97% | 4.2–4.9 |

- With many labels, fine-tuned is as accurate as pretrained. The class information survives fine-tuning.
- But fine-tuned embeddings put 90% of their variance in five directions, exactly what a six-class head needs. The rest of the information sits in weak directions that a few labels cannot find.
- Untrained embeddings also use few directions, but stay weak with many labels. Their information is missing.
- This is not a loading bug. Loaded the way the experiment loads them, the fine-tuned models reproduce their 6-class test accuracy (99.85–99.97%).

## Models (F3)

Paired differences on the same test episodes (pp), pretrained baseline. "n.s." means the 95% CI includes 0.

| Comparison | 1-shot | 5-shot | 10-shot |
|---|---|---|---|
| DINO − ViT-B/16 | +6.8 | +5.5 | +3.5 |
| ConvNeXt − ResNet | −4.6 | −1.3 | 0.0 (n.s.) |
| Transformers − CNNs | −1.3 | −0.5 | 0.0 (n.s.) |

- At 1 shot, ResNet (64.6) and DINO (64.4) tie for best.
- ViT-B/16 is the weakest pretrained model. "Transformers − CNNs" averages a strong DINO with a weak ViT, so the two cancel out.
- After fine-tuning, the Transformers keep more: Transformers − CNNs is +3.6, +6.4 and +5.9 pp.

## Augmentation (F6, F8, T3, F9)

| Baseline | Gain with all 8 methods (pp) | Share of the 1→5-shot gap closed | Best single methods |
|---|---|---|---|
| Untrained | −0.2 to +3.4 | 8–18% | brightness/contrast |
| Pretrained | +0.6 to +1.9 | 4–8% | rotation (90°) |
| Fine-tuned | +4.4 to +8.6 | 21–27% | blur, scale up |

- The repeat controls (same number of rows, no new pixels) stay near 0. The gains come from new pixels, not from more rows.
- The method chosen on development episodes gives a clear test gain in 31 of 36 cases (T3). The other five are untrained ConvNeXt (1, 5 shots), untrained DINO (10 shots) and pretrained DINO (1, 5 shots).
- Pretrained models pick rotation in 11 of 12 cases. DINO gains the least.
- Fine-tuned models gain most from blur (+3.3 pp on average) and scale up (+2.6 pp).
- More views (F9, 1 shot):
  - Rotation keeps helping pretrained models.
  - Brightness/contrast keeps helping untrained and fine-tuned models (fine-tuned: +0.4 pp with 1 view, +3.2 pp with 8).
  - For fine-tuned models, blur and scale up level off.
  - For pretrained models, noise and blur get worse with more views (noise: −0.6 pp with 1 view, −2.3 pp with 8).
- Gains are averages. In 26–66% of test episodes, the chosen method gives no gain.

## Checks

- All 720 result groups and all 16 figures and tables were made. No classifier fit hit the iteration limit.
- C was picked at an end of its range in 5 of 720 groups. All are 1-shot, where accuracy barely changes with C.
- **One artifact:** untrained ViT-B/16 at 5 shots, with no augmentation.
  - Its development episodes picked a C that did worse on test: 54.9%, against 56.3% in an earlier run with a narrower C range.
  - So every augmentation gain in that column looks about 1 pp too large.
  - The repeat×2 control shows it: +1.2 pp instead of 0. Read that column with care.

## What we cannot claim

- **Small gaps inside one baseline.** Every random choice uses one seed: one untrained draw, one fine-tuning run per learning rate, and one view per image and method. The CIs cover episodes only. So small gaps inside the untrained or fine-tuned baseline are not reliable. One example is fine-tuned DINO vs ViT at 5 and 10 shots (about 0).
- **These untested ideas.** They are not findings:
  - Fine-tuned features may lean on fine texture from the six fine-tuning classes, and blurred views may teach the classifier to ignore it.
  - Why noise hurts pretrained models.
  - Why ViT-B/16 is weak. Training on ImageNet-1K alone gives a weak ViT, and the 64 px tiles are upsampled to 224 px.

## For discussion

1. Agree to replace DINOv2 with DINO ViT-S/16 (research_info.md, section 8).
2. Choose the main-text figures and tables. The plan has eight: T1, T5, F2, F3, F11, F6, F8 and T3.
