# EuroSAT Representation Benchmark

CS7643 research project comparing frozen CNN and Transformer representations with 1, 5 and 10 labeled images per land-use class. The [proposal](docs/project-proposal.pdf) defines the study; [TODO.md](TODO.md) tracks implementation and experiments for four teammates.

![EuroSAT class examples](docs/figures/class_examples.png)

Native RGB support images from EuroSAT (Helber et al., 2019; Copernicus Sentinel-2). [Dataset audit and figure provenance](docs/dataset.md).

## Run in Colab

1. Open the notebook linked below; it already includes the code and saved outputs.
2. Upload `EuroSAT_Representation_Benchmark.ipynb` to Colab and choose a **GPU** runtime.
3. Run all cells and authorize Drive. The default `MODE = 'full'` runs the complete ResNet-18 and DINOv2 suite.
4. Use `MODE = 'smoke'` only for a quick execution check. Each teammate can change `ASSIGNMENT` for other workstreams.

In VS Code, choose **Select Another Kernel → Colab → New Colab Server → GPU → Python 3 (ipykernel)**. The runtime must have an NVIDIA GPU; selecting Python on a TPU server does not change its hardware. The notebook checks the device before downloading data. Colab allocation and account authorization happen in your own account.

Open **[EuroSAT_Representation_Benchmark.ipynb](EuroSAT_Representation_Benchmark.ipynb)** for all notebook work. It contains the actual helper code in twelve labeled, readable cells and needs no GitHub credentials when uploaded to Colab. Saved dataset previews are visible before execution, followed by dataset tables, six figures, four labeled implementation parts with TODO checkboxes, and one selected execution cell. The same file works locally in VS Code with `TARGET = 'local'`.

## Local development

Use Python 3.11 and run these commands from the repository root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/local.txt
python scripts/build_notebook.py
python -m unittest discover -s tests -v
python -m eurosat run --mode smoke
```

Select `.venv` as the VS Code kernel for local work and set notebook `TARGET = 'local'`. Local runs use available CUDA, Apple MPS or CPU. Keep Colab's existing CUDA/PyTorch installation; `requirements/local.txt` is for local setup.

```sh
python -m eurosat explore --output docs/figures
python scripts/execute_notebook.py --mode full
python -m eurosat merge outputs/model-a/results.json outputs/model-b/results.json --output outputs/combined
python -m eurosat report outputs/combined/results.json
```

## Repository structure

Start with the notebook to run experiments. Edit `src/eurosat/` to change reusable behavior; rebuild the notebook afterward so Colab runs the same code.

```text
eurosat-representation-benchmark/
├── EuroSAT_Representation_Benchmark.ipynb  Run, inspect outputs, follow Parts A–D
├── src/eurosat/                           Reusable experiment implementation
├── scripts/                              Build and validate the notebook
├── tests/                                Reproducibility, leakage and recovery checks
├── requirements/local.txt                Local environment dependencies
├── docs/                                 Proposal, methods, figures and team guide
├── TODO.md                               Shared checkoff plan
└── .github/workflows/ci.yml               Automated checks on push and pull requests
```

| To change… | Edit here | Notebook location / owner |
|---|---|---|
| Run mode, models, seeds and view budgets | `src/eurosat/config.py` | Sections 1 and 4; shared protocol |
| GPU selection and device checks | `src/eurosat/runtime.py` | Section 3: hardware helpers |
| Downloading, duplicate removal, splits and shot sampling | `src/eurosat/data.py` | Part A: shared data / ResNet |
| Dataset audit and sample-image plots | `src/eurosat/exploration.py` | Section 5; Part A |
| Pretrained backbone adapters | `src/eurosat/models.py` | Parts A/B/C/D: ResNet / ConvNeXt / ViT / DINOv2 |
| Support-image transformations | `src/eurosat/augment.py` | Parts B and C |
| Batched embeddings and interruption recovery | `src/eurosat/features.py`, `cache.py` | Shared infrastructure |
| Logistic regression, metrics and paired summaries | `src/eurosat/evaluation.py` | Part D |
| Merge teammate exports and generate result figures | `src/eurosat/reporting.py` | Part D; Section 8 |
| Execution order and cache reuse | `src/eurosat/pipeline.py` | Section 7 |
| Complete DINOv2 experiment suite | `src/eurosat/experiments.py` | Part D: strength selection, main grid and view budgets |
| Terminal commands | `src/eurosat/cli.py` | `python -m eurosat --help` |
| Notebook headings, TODO cells and execution checks | `scripts/build_notebook.py`, `execute_notebook.py` | Development tooling |

`data/`, `cache/`, `outputs/`, `work/` and `.venv/` are generated locally or in Drive and stay outside Git. The notebook's reviewed tables and figures stay saved for readers.

The executor saves cell outputs into this same notebook; it does not create another notebook copy. Rebuilding preserves saved outputs when the code is unchanged; changed code starts with measured dataset previews.

Edit reusable behavior in `src/eurosat`, and notebook narrative in `scripts/build_notebook.py`; rebuild and commit the tracked notebook. `dist/`, environments, datasets, caches and raw result exports are ignored; reviewed notebook outputs are retained. CI checks the notebook, runs unit tests and builds the package.

## Metrics and baselines

We report accuracy, Macro-F1, macro precision/recall, balanced accuracy, weighted F1, per-class precision/recall/F1 and query counts, confusion matrices, MCC and Cohen's kappa. Paired augmentation/model/budget differences have trial SDs and conditional 95% confidence intervals when enough trials are available. Time, memory, model size and cache usage are recorded separately.

The proposal's baselines are explicit: matched **ResNet-18 versus DINOv2**, no augmentation, repeated-original controls, and 5-/10-original label budgets. Model and augmentation selection still use development accuracy only. See [metric definitions and uncertainty limits](docs/metrics.md).

## Part D: DINOv2 and result visualization

Part D includes the full proposal workflow and the matched ResNet-18 comparison baseline. The notebook defaults to `ASSIGNMENT = 'dinov2_vits14'`; use another assignment for Parts A–C. DINOv2 uses the official pinned ViT-S/14 checkpoint, frozen evaluation mode and the same development-tuned logistic regression as the other backbones.

The suite selects strengths on development data, runs every four-view condition, then evaluates development-selected nine-view methods on the same 1-shot originals. Section 8 shows accuracy, Macro-F1, class effects, confusion matrices and the proposal's label/view-budget comparisons. Error bars show trial standard deviation. Every plot is labeled when it comes from smoke mode. The saved notebook now includes the completed full-data Colab T4 run: 600 DINOv2 records, 600 matched ResNet-18 records and 80 nine-view follow-ups. ConvNeXt/ViT integration and independent report review remain on the TODO list.

```sh
# Quick DINOv2 execution check (three default conditions).
python -m eurosat part-d --mode smoke
# Full Part D suite; run on the intended GPU after freezing the protocol.
python -m eurosat part-d --mode full
# Generate PNG/PDF figures, CSVs and provenance from a completed export.
python -m eurosat report outputs/EXPERIMENT_ID/results.json
```

The report exports all prespecified conditions. It selects the preferred policy using development accuracy only; test accuracy never chooses the policy. Timing exports describe recorded extraction stages and classifier fits, including possible cache reuse. See [Part D implementation and interpretation](docs/part-d.md).

## Study and recovery

The primary study uses all ten classes, deduplicated 60/20/20 support/development/test partitions, frozen ResNet-18, ConvNeXt-Tiny, ViT-B/16 and DINOv2, and the same development-tuned logistic regression. Full mode includes 20 paired trials, ten augmentation/control conditions and three shot budgets: 2,400 classifier results. Smoke scores are execution checks, not research findings.

Colab saves the archive, weights, feature chunks, classifier checkpoints and completed trial records to `MyDrive/ColabNotebooks/cs7643/EuroSATRepresentationBenchmark`. Raw images are extracted to disposable runtime-local storage. Rerunning resumes matching cached work; a restart may need to extract images again. Local caches and Drive caches are separate. Changing relevant settings creates different cache identities; increment `PIPELINE_VERSION` for semantic code changes.

Read [collaboration](docs/collaboration.md) for ownership, branches, shared Drive and merging; [methodology](docs/methodology.md) for protocol and limitations; [original-notebook migration](docs/migration.md) for what was retained or replaced; and [validation](docs/validation.md) for execution evidence.

Existing Drive storage named `EuroSATLowData` is reused automatically to preserve earlier caches. New storage uses `EuroSATRepresentationBenchmark`; set `EUROSAT_STORAGE` to override it.
