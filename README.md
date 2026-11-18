# EuroSAT Few-Shot Benchmark

CS7643 project. Four image models (ResNet-18, ConvNeXt-Tiny, ViT-B/16 and DINO ViT-S/16) classify satellite land cover from 1, 5 or 10 labeled images per class. Each model is tested untrained, ImageNet-pretrained and fine-tuned, with and without augmenting the labeled images.

**Team:** Andrew Dao, Eric Sheng, Eunhyeok Jang and Yanru Yang.

## Start here

| File | What it is |
|---|---|
| [research_info.md](research_info.md) | The plan: design, setup, progress and limitations |
| [figure_n_table_sug.md](figure_n_table_sug.md) | The 16 figures and tables, and what each one shows |
| [EuroSAT_Representation_Benchmark.ipynb](EuroSAT_Representation_Benchmark.ipynb) | Runs everything, from the data check to the report |
| [SPLITS.md](SPLITS.md) | The fixed data split |
| [docs/project-proposal.pdf](docs/project-proposal.pdf) | The original proposal. research_info.md lists what changed |

## Run it locally

Tested on Windows with an RTX 5060 Ti, Python 3.12:

```sh
conda create -n eurosat python=3.12
conda activate eurosat
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements/local.txt
jupyter notebook EuroSAT_Representation_Benchmark.ipynb
```

In the notebook, set `MODE` in Section 1 (`'smoke'` for a quick code check, `'full'` for the experiment). Then choose **Kernel → Restart Kernel and Run All Cells**.

- The first run downloads EuroSAT (90 MB) and the pretrained weights (0.6 GB).
- On the RTX 5060 Ti, fine-tuning takes about 3 hours. The rest takes about 2 hours the first time, while embeddings are computed.
- Saved work is reused. A stopped run resumes, and finished models are not trained again.
- Restart the kernel after you change code in `src/`.

From a terminal, the same steps are:

```sh
python -m eurosat finetune --models all
python -m eurosat run --mode full
```

The terminal run does not make the report. Run the notebook afterwards: it reuses the saved work and makes the figures and tables.

## Run it on Colab

1. Choose a GPU runtime and set `TARGET = 'colab_gpu'` in Section 1.
2. Storage goes to Google Drive, by default `MyDrive/ColabNotebooks/cs7643/EuroSATRepresentationBenchmark`. To use another folder, such as a shared one, set `STORAGE_DIR`.
3. The notebook needs the code: set `CODE_DIR` to a checkout of this repository, or copy `src/` to `<storage>/code/src/`.
4. Copy `splits_manifest.json` into the storage folder, or set `MANIFEST_FILE`.

## Where files go

All of these stay outside Git.

```text
data/             EuroSAT.zip and the extracted images
cache/
  embeddings/     embeddings for every backbone, image and view
  results/        one file per result group
  weights/        pretrained downloads and fine-tuned models
  costs/          embedding cost per backbone (for T2)
  data-audits/    data-check exports
outputs/<run>/    results.json and report/ (figures as PNG and PDF, tables as CSV)
```

## Code map

```text
src/eurosat/
  config.py         settings and defaults
  notebook.py       what each notebook section runs
  cli.py            terminal commands
  preprocessing/    data download and checks, the split, image transforms and augmentation
  models/           one file per model, and how the three baselines are built
  training/         fine-tuning
  experiments/      embeddings, classifier, conditions, the run loop, the many-label check, results
  reporting/        figures, tables and the report
scripts/
  build_notebook.py     writes the notebook from its cell text
  execute_notebook.py   runs the notebook in a fresh kernel
make_splits.py      made splits_manifest.json (kept for reference)
```

## Working together

- Use feature branches and pull requests.
- Keep the shared `splits_manifest.json`. Do not make new splits.
- To change notebook cells, edit `scripts/build_notebook.py` and run `python scripts/build_notebook.py`.
- Only one run should write to a storage folder at a time.
- Smoke results only check the code. They are not research results.

## Data

EuroSAT: Patrick Helber, Benjamin Bischke, Andreas Dengel and Damian Borth, "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," IEEE JSTARS, 2019. Sentinel-2 imagery: Copernicus. See the [official repository](https://github.com/phelber/EuroSAT) for the citation and terms.
