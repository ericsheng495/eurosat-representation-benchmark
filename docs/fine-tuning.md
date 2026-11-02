# DINOv2 partial fine-tuning extension

Use [EuroSAT_Fine_Tuning.ipynb](../EuroSAT_Fine_Tuning.ipynb). This is a separate exploratory extension of the [proposal](project-proposal.pdf), whose page 4 explicitly allows fine-tuning if resources permit. The main four-model frozen-backbone benchmark and its saved results remain the primary study.

## Question and decision

Can adapting DINOv2's representation improve few-shot satellite-image classification more than support-image augmentation alone? The reported small augmentation gains and larger benefits from extra real labels motivate testing representation adaptation. They do not imply that adaptation will succeed.

**Selected direction:** official pinned DINOv2 ViT-S/14; train the last Transformer block, final LayerNorm and a new ten-class linear head on L2-normalized class-token embeddings. Freeze the patch embedding, positional/class tokens and earlier blocks. This tests whether late representations benefit from satellite-specific adaptation while updating fewer weights than deeper adaptation. It is a research hypothesis, not an established best setting.

`LAST_BLOCKS = 1` is the default. Setting it to `2` runs a separately cached two-block sensitivity study. Never select the scope from held-out query scores. LoRA/adapters and normalization-only training are possible future directions, not implemented experiments. Full-backbone training is disabled.

## Proposal alignment

| Proposal section | Preserved commitment |
|---|---|
| Approach, p. 1 | Same official DINOv2 checkpoint and recorded software |
| Datasets, p. 3 | Ten EuroSAT classes, verified/deduplicated data, 224-pixel preprocessing, 60/20/20 partitions |
| Evaluation, p. 3–4 | Identical original support/query images and nested 1/5/10 shots; development labels disclosed separately |
| Optional extension, p. 4 | Fine-tuning reported separately from frozen representations |
| Augmentation, p. 4–5 | Support-only transformations, matched repeated originals and original-plus-four-view comparison |
| Performance/cost, p. 5 | Accuracy, Macro-F1, class recall, confusion matrices, paired gains, parameter counts and timing |

The primary study uses scikit-learn L2-regularized multinomial logistic regression. This extension uses a neural linear head trained jointly with selected backbone weights. Its matched baseline trains the identical head with a frozen backbone. Comparing these two conditions holds classifier and optimizer rules fixed. Direct differences against the primary notebook also change the classifier training procedure, so cannot isolate adaptation alone.

## Prespecified experiments

Cross both backbone conditions with all three support treatments at each 1/5/10-shot budget:

| Backbone condition | No augmentation | Repeated originals | Flip augmentation |
|---|---|---|---|
| Frozen DINOv2 + trained head | Baseline | Exposure/control baseline | Augmentation-only baseline |
| Last-block DINOv2 + trained head | Adaptation alone | Adaptation with matched repeated exposures | Adaptation + augmentation |

Flip is a proposal-listed geometric policy fixed before this extension, not chosen from query performance. In full mode, repeat and flip both add four rows per original. Originals and fixed views are shared across backbone conditions. Repeats matter here because extra optimizer steps can affect a neural classifier even when image content is unchanged. This focused extension does not replace the primary notebook's broader policy grid or selected nine-view experiments.

Report paired accuracy and Macro-F1 gains for adaptation versus the frozen head within each treatment, flip versus no augmentation, flip versus repeats, and 5/10 real originals versus 1 original. Full mode additionally compares 1 original + 4 flips against 5 originals. These match input-row count, not independent-label information or total compute. No 1+4-versus-5 claim is emitted from a one-view smoke run.

## Settings and budget

`MODE = 'smoke'`: 40 real images/class, one trial, two epochs, one added view; 18 fits. `MODE = 'full'`: all images, 20 trials, up to ten epochs, four added views; 360 fits. Mode changes dataset/trial/view budgets, never the scope of unfrozen weights. The two-block ablation is a separate additional budget.

Both conditions use AdamW, head learning rate 0.001, backbone learning rate 0.00001, weight decay 0.0001, matched head initialization and batch order. These are starting settings, not tuned optima. Select any revisions on development data and lock the protocol before final held-out evaluation. Each epoch measures development accuracy; highest accuracy selects weights and lower loss breaks ties. Early stopping permits three epochs without improvement. Backbone dropout and pretrained normalization buffers stay in evaluation mode; gradients still update selected parameters.

Only sampled support originals and their views enter optimization. Development labels are additional selection information outside the nominal shot count. Query labels are used after best weights are restored. This extension is exploratory because earlier query results motivated it; reusing those partitions does not create an independent confirmation set. Image-level partitioning does not establish geographic independence.

## Recovery, outputs and validation

Data archives and pretrained weight downloads are shared with the primary benchmark. Updated backbone weights invalidate frozen embedding caches, so training uses images directly. Each epoch atomically saves only trainable parameters, optimizer state, best weights, history and the next epoch under `cache/finetuning/<id>/`; frozen weights reload from the pinned pretrained checkpoint. Tensor-only `torch.load(..., weights_only=True)` is used. A restart repeats at most the incomplete epoch, and completed jobs skip training. Configuration IDs distinguish model, block count, policy, view count, dataset, trials and software.

Each treatment has separate resumable artifacts. The study combines results under `outputs/finetuning/<study-id>/` with JSON manifests and full classification metrics, trial/summary CSV, `paired-contrasts.csv`, and PNG/PDF comparison and learning-curve figures. Report trainable/total parameters, selected epochs, timing and device alongside scores. Single-trial smoke outputs verify execution, not statistical evidence of a benefit.

Tests cover trainable scopes, one/two-block selection, matched initialization, exact deterministic interrupted/resumed training, query evaluation placement, repeated controls, paired effect arithmetic and notebook/source consistency. Run the notebook in smoke mode before committing GPU time to the full grid.

Sources: [DINOv2 paper](https://arxiv.org/abs/2304.07193), [official DINOv2 code](https://github.com/facebookresearch/dinov2). These describe the pretrained representation; the proposed EuroSAT adaptation and its expected benefits must be tested here.

Executed validation: all 18 one-block DINOv2 smoke fits completed locally on Apple MPS; the notebook retains three real result figures plus its support-image preview. The one-block condition trains 1,779,850 of 22,060,426 parameters (about 8.1%); the matched head-only control trains 3,850. A fresh-kernel rerun reused all completed jobs. The full 360-fit grid has not been run.

## Code organization

`src/eurosat/finetuning.py` owns settings, model adaptation, training and experiment orchestration. `src/eurosat/finetuning_reporting.py` builds tables once for CSV exports, notebook displays and figures. Both reuse shared data, preprocessing, checkpoint and metric functions.

The Colab notebook embeds actual code generated by `scripts/build_finetuning_notebook.py`. It includes only needed declarations and their dependencies, consolidates imports, and separates fine-tuning settings, model, training/recovery and execution into readable cells. Unused frozen-feature extraction and dataset-exploration functions are omitted. Run `python scripts/build_finetuning_notebook.py` after source edits; `--check` detects notebook drift. This cleanup retains existing experiment IDs, checkpoints and scores.
