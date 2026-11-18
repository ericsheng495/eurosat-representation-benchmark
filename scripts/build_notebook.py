"""Generate EuroSAT_Representation_Benchmark.ipynb. Cells stay short: each section is one call into src/eurosat/notebook.py.

Edit the cells here, then run `python scripts/build_notebook.py` (`--check` only compares).
"""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'EuroSAT_Representation_Benchmark.ipynb'


def load_config_module():
    """Read config.py directly so building the notebook does not require torch."""
    spec = importlib.util.spec_from_file_location('eurosat_config', ROOT / 'src/eurosat/config.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def settings_cell():
    """The one cell people edit. Augmentation defaults are written in from config.py."""
    config = load_config_module()
    defaults = config.AUGMENTATION_DEFAULTS
    plain = ', '.join(f'{m!r}: {{}}' for m, p in defaults.items() if not p)
    strengths = '\n'.join(f'    {m!r}: {p!r},' for m, p in defaults.items() if p)
    return f'''# Where to run
TARGET = 'local'        # 'local' or 'colab_gpu'
STORAGE_DIR = ''        # data, cache, weights and results ('' = this repository, or Google Drive on Colab)
CODE_DIR = ''           # folder that contains src/ ('' = find it automatically)
MANIFEST_FILE = ''      # splits_manifest.json ('' = <storage>/splits_manifest.json)

# What to run
MODE = 'smoke'          # 'smoke' = quick code check, 'full' = the experiment
MODELS = 'all'          # or a subset, e.g. 'resnet18,dino_vits16'
BASELINES = {config.BASELINES!r}
TRACKS = [1, 2, 3]      # 1 models, 2 all augmentations together, 3 each augmentation alone
VIEW_SWEEP = True       # extra 1-shot run with 1, 2, 4 and 8 augmented views (figure F9)

# Augmentation strength on the 224x224 input (defaults: src/eurosat/config.py)
AUGMENTATION = {{
    {plain},  # no strength setting
{strengths}
}}

# Display
SHOW_DETAILS = False    # also show every data-check table and the supplementary figures and tables
REPORT_RESULTS = ''     # path to a saved results.json: show its results without computing anything
'''


SETUP = '''import os
import sys
from pathlib import Path

# Automated runs (scripts/execute_notebook.py) can override these with EUROSAT_* variables.
TARGET = os.environ.get('EUROSAT_TARGET', TARGET)
STORAGE_DIR = os.environ.get('EUROSAT_STORAGE', STORAGE_DIR)
CODE_DIR = os.environ.get('EUROSAT_CODE_DIR', CODE_DIR)

if TARGET == 'colab_gpu':  # Storage on Google Drive; images are extracted to Colab's fast local disk.
    from google.colab import drive
    drive.mount('/content/drive')
    parent = Path('/content/drive/MyDrive/ColabNotebooks/cs7643')
    STORAGE = Path(STORAGE_DIR or next((p for p in [parent / 'EuroSATLowData'] if p.exists()),
                                       parent / 'EuroSATRepresentationBenchmark'))
    if STORAGE_DIR and not STORAGE.is_dir():
        raise FileNotFoundError('STORAGE_DIR not found on Google Drive: check the My Drive shortcut')
    SCRATCH = Path('/content/eurosat-representation-benchmark')
else:  # Local: everything under the repository unless STORAGE_DIR says otherwise.
    here = Path.cwd()
    STORAGE = Path(STORAGE_DIR or next((p for p in [here, *here.parents] if (p / 'pyproject.toml').exists()), here))
    SCRATCH = STORAGE / 'data/extracted'
STORAGE.mkdir(parents=True, exist_ok=True)

places = [Path(CODE_DIR)] if CODE_DIR else [Path.cwd(), *Path.cwd().parents, Path('/content/eurosat-representation-benchmark'),
                                           STORAGE / 'code']
code = next((p for p in places if (p / 'src/eurosat/__init__.py').is_file()), None)
if code is None:
    raise FileNotFoundError('Project code not found: set CODE_DIR to the repository, or copy src/ into <storage>/code/')
sys.path.insert(0, str(code / 'src'))

from eurosat import notebook as nb
run = nb.Run.from_settings(globals())
'''

OVERVIEW = '''# EuroSAT Few-Shot Benchmark

Four image models (ResNet-18, ConvNeXt-Tiny, ViT-B/16, DINO ViT-S/16), each **untrained**, **ImageNet-pretrained** and **fine-tuned**, classify satellite land cover from 1, 5 or 10 labeled images per class, with and without augmenting those images. Plan: `research_info.md`.

**How to use:** edit Section 1, then run all cells. Start with `MODE = 'smoke'`.
'''

SECTIONS = [
    ('1 · Settings', 'The only cell to edit.', None),
    ('2 · Setup', 'Connects storage, loads the project code and shows what this run will do.', SETUP),
    ('3 · Data check', 'Verifies the dataset and selects the fixed episodes.', 'run.check_data()'),
    ('4 · Fine-tune', 'Trains the fine-tuned baseline. Models already trained are skipped.', 'run.fine_tune()'),
    ('5 · Run', 'Embeddings and classifiers for every baseline × model × condition, then the report figures and '
                'tables. Saved work is reused, so a stopped run resumes.', 'run.run()'),
    ('6 · Results: models and baselines', 'T1, F2, F3, F11 (figure_n_table_sug.md).', 'run.show_models()'),
    ('7 · Results: augmentation', 'F6, F8, T3.', 'run.show_augmentation()'),
    ('8 · Checks and files', 'Classifier checks, labels used (T4) and where the outputs are saved.', 'run.show_checks()'),
]


def build(*, write=True):
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md(OVERVIEW)]
    for title, text, source in SECTIONS:
        cells += [md(f'## {title}\n{text}'), code(settings_cell() if source is None else source.strip() + '\n')]
    for i, cell in enumerate(cells):
        cell.id = 'cell-' + hashlib.sha256(f'{i}:{cell.source}'.encode()).hexdigest()[:12]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU', 'colab': {'gpuType': 'T4', 'provenance': []}})
    if NOTEBOOK.exists():  # Keep earlier outputs when the code cells did not change.
        old = [c for c in nbf.read(NOTEBOOK, 4).cells if c.cell_type == 'code']
        new = [c for c in notebook.cells if c.cell_type == 'code']
        if [c.source for c in old] == [c.source for c in new]:
            for before, after in zip(old, new):
                after.outputs, after.execution_count = before.outputs, before.execution_count
    nbf.validate(notebook)
    if write:
        nbf.write(notebook, NOTEBOOK)
    return notebook


def canonical_cells(notebook):
    return [(c.cell_type, c.source, {k: v for k, v in c.metadata.items() if k != 'execution'}) for c in notebook.cells]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    notebook = build(write=not args.check)
    if args.check and canonical_cells(notebook) != canonical_cells(nbf.read(NOTEBOOK, 4)):
        raise SystemExit('Notebook is stale; run python scripts/build_notebook.py')
    print(NOTEBOOK)


if __name__ == '__main__':
    main()
