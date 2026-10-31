# Part D: DINOv2 and reporting

## Implemented workflow

1. Choose `ASSIGNMENT = 'dinov2_vits14'` at the notebook top. `MODE = 'smoke'` checks execution; `MODE = 'full'` selects all ten conditions, 1/5/10 shots, 20 paired trials and four added support views.
2. Run the notebook in order. The actual adapter appears in section 3's backbone cell. It loads the official DINOv2 ViT-S/14 source at revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, uses its class-token representation, disables gradients and enters evaluation mode. Original and transformed features use the shared chunked cache and L2 normalization.
3. The common classifier selects logistic-regression C on development accuracy. Test labels only measure held-out accuracy, Macro-F1, class recall and confusion counts.
4. Section 8 calls `create_report` on the completed `results.json`. No additional model inference is needed for plotting. The terminal equivalent is `python -m eurosat report PATH/TO/results.json`.

## Outputs

| File | Interpretation |
|---|---|
| `model_by_shot.png/.pdf` | Unaugmented held-out accuracy by original shots, mean ± trial SD |
| `augmentation_gains.png/.pdf` | Every non-baseline condition's paired gain, mean ± paired-trial SD |
| `class_recall.png/.pdf` | Per-class recall for the highest requested shot budget, unaugmented baseline |
| `confusion_matrix.png/.pdf` | Highest-shot baseline confusion counts summed over trials, normalized by true class |
| `summary.csv` | Mean metrics and paired gains for all prespecified conditions |
| `paired-trials.csv` | Trial-level scores and paired gains, preserving trial/model/shot identity |
| `development-selected-policies.json` | One policy per model/shot selected only by development performance |
| `timings.csv` | Mean classifier-fit time and range of recorded extraction-stage times |
| `provenance.json` | Dataset/configuration/software/checkpoints, source file hash and available execution records |
| `report.json` | Run mode, result count, figure names and variability definition |

The same function handles merged compatible model exports, so Part D can integrate Parts A–C without a different plotting protocol. Incomplete or duplicated trial grids and unsupported checkpoint IDs are rejected before rendering.

## Interpretation limits

Standard deviation across trials is not a confidence interval. The same held-out queries appear repeatedly; summed confusion counts do not represent additional independent images. Class-level plots use a fixed baseline rather than selecting the most favorable test-scoring augmentation. Development labels are additional labeled selection data beyond the original-shot budget.

Extraction times can include model loading or warm-cache reads and are repeated in trial records. Do not sum those values or present them as isolated GPU throughput. Execution records describe sessions, which may reuse older computed results; they do not prove the device used for every cached trial. Geographic separation is not established by the image-level split. DINOv2 versus supervised ViT differences also include pretraining data, architecture and model size, so the comparison does not isolate self-supervision.

Full research grids, four-model integration, independent review and final paper interpretation remain unchecked in TODO.md until actually completed.
