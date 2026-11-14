"""Generate the benchmark notebook. Notebook cells stay short; computation lives in src/eurosat/.

Edit the cells here, then run `python scripts/build_notebook.py`. Section helpers are in
src/eurosat/notebook.py.
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


def canonical_cells(notebook):
    return [(c.cell_type, c.source, {k: v for k, v in c.metadata.items() if k != 'execution'})
            for c in notebook.cells]


def section(title, does, get, edit):
    """Every section opens with the same three short lines."""
    return f'## {title}\n\n- **Does:** {does}\n- **You get:** {get}\n- **Edit:** {edit}\n'


OVERVIEW = '''# EuroSAT Few-Shot Benchmark

Compares four frozen pretrained backbones (ResNet-18, ConvNeXt-Tiny, ViT-B/16 and DINOv2 ViT-S/14) on fixed 4-way few-shot episodes from EuroSAT, with 1, 5 and 10 labeled images per class and optional augmentation of those labeled images.

`satellite image → (support augmentation) → frozen backbone → embedding → logistic regression → prediction`

**How to use**
1. Edit **Section 1 · Settings**. It is the only cell you need to change.
2. Run all cells in order. Start with `MODE = 'smoke'` to check that everything works.
3. Read the results in Sections 5 and 6.

| Section | What it does | Edit? |
|---|---|---|
| 1 · Settings | Hardware, run size, models, augmentation | **Yes** |
| 2 · Setup | Installs packages, connects storage, loads the project code | Only if your folders differ |
| 3 · Data check | Verifies the dataset and the fixed episodes | No |
| 4 · Run | Extracts embeddings and fits the classifiers (cached) | No |
| 5 · Results: models | Accuracy, comparison with ResNet-18, per-model details | No |
| 6 · Results: augmentation | Gains from support augmentation, if enabled | No |
| 7 · Saved files | Where results and figures are stored | No |

Smoke runs only check that the code runs; they are not research results. Background: `research_info.md` (plan), `figure_n_table_sug.md` (figures), `docs/` (methods).
'''


def settings_cell():
    """The single cell users edit. Augmentation defaults are written in from config.py."""
    config = load_config_module()
    magnitudes = '\n'.join(f'    # {config.AUGMENTATION_NOTES[policy]}\n    {policy!r}: {params!r},'
                           for policy, params in config.AUGMENTATION_DEFAULTS.items())
    available = ', '.join(config.POLICIES + config.OPTIONAL_POLICIES)
    return f'''# -- Where to run ---------------------------------------------------------------
TARGET = 'colab_gpu'   # 'colab_gpu' = Colab GPU runtime, 'local' = this computer
STORAGE_DIR = ''       # Shared folder for data, cache and results ('' = default, see Section 2)
CODE_DIR = ''          # Repository folder that contains src/ ('' = search automatically)
MANIFEST_FILE = ''     # Path to splits_manifest.json ('' = <storage>/splits_manifest.json)

# -- What to run ----------------------------------------------------------------
MODE = 'smoke'         # 'smoke' = 2 episodes per shot (quick check), 'full' = 600 test episodes per shot
MODELS = 'all'         # 'all', or a comma-separated subset: resnet18, convnext_tiny, vit_b_16, dinov2_vits14

# -- Augmentation of the labeled support images ---------------------------------
# 'none' (no augmentation) is required: every other condition is compared with it.
# 'repeat' copies each original (VIEWS + 1) times: same number of rows, no new pixels.
# Available: {available}
POLICIES = ['none']
VIEWS = 4              # Augmented views added per original support image

# Strength of each method on the 224x224 image. Each view draws a new random value
# within these limits. Defaults: AUGMENTATION_DEFAULTS in src/eurosat/config.py.
AUGMENTATION = {{
{magnitudes}
}}

# -- Display and report-only mode -----------------------------------------------
SHOW_DETAILS = False   # True = show every data-check table and extra result figures
REPORT_RESULTS = ''    # Path to a saved results.json: redraw results without data or GPU
'''


BOOTSTRAP = '''import os
import sys
import subprocess
import importlib.util
from pathlib import Path

# Automated runs (scripts/execute_notebook.py) pass settings as EUROSAT_* environment variables.
TARGET = os.environ.get('EUROSAT_TARGET', TARGET)
STORAGE_DIR = os.environ.get('EUROSAT_STORAGE', STORAGE_DIR)
CODE_DIR = os.environ.get('EUROSAT_CODE_DIR', CODE_DIR)
if TARGET not in {'colab_gpu', 'local'}:
    raise ValueError("TARGET must be 'colab_gpu' or 'local'")

# 1) Install any missing packages.
requirements = {'torch': 'torch>=2.6,<3', 'torchvision': 'torchvision>=0.21,<1',
                'numpy': 'numpy>=1.26,<3', 'sklearn': 'scikit-learn>=1.4,<1.8',
                'PIL': 'Pillow>=10,<13', 'pandas': 'pandas>=2.2,<3',
                'matplotlib': 'matplotlib>=3.8,<4', 'seaborn': 'seaborn>=0.13,<0.14'}
missing = [package for module, package in requirements.items() if importlib.util.find_spec(module) is None]
if missing:
    print('Installing:', ', '.join(missing))
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', *missing])

# 2) Persistent storage: Google Drive on Colab, the repository folder locally.
if TARGET == 'colab_gpu':
    from google.colab import drive
    drive.mount('/content/drive')
    parent = Path('/content/drive/MyDrive/ColabNotebooks/cs7643')
    default = parent / 'EuroSATLowData' if (parent / 'EuroSATLowData').exists() else parent / 'EuroSATRepresentationBenchmark'
    STORAGE = Path(STORAGE_DIR or default)
    if STORAGE_DIR and not STORAGE.is_dir():
        raise FileNotFoundError('Shared storage not found: check the My Drive shortcut and STORAGE_DIR')
    SCRATCH = Path('/content/eurosat-representation-benchmark')  # Extracted images on the fast local disk
else:
    here = Path.cwd()
    project = next((p for p in [here, *here.parents] if (p / 'pyproject.toml').exists()), here)
    STORAGE = Path(STORAGE_DIR or project)
    SCRATCH = STORAGE / 'data/extracted'
STORAGE.mkdir(parents=True, exist_ok=True)

# 3) Project code: a repository checkout, or <storage>/code/src on Colab.
candidates = ([Path(CODE_DIR)] if CODE_DIR else
              [Path.cwd(), *Path.cwd().parents, Path('/content/eurosat-representation-benchmark'), STORAGE / 'code'])
source_root = next((p for p in candidates if (p / 'src/eurosat/__init__.py').is_file()), None)
if source_root is None:
    raise FileNotFoundError('Project code not found: set CODE_DIR, or copy src/ into <storage>/code/src/')
sys.path.insert(0, str(source_root / 'src'))

print('Shared storage :', STORAGE)
print('Local images   :', SCRATCH)
print('Project code   :', source_root)
'''


CONFIGURE = '''from IPython.display import display
from eurosat import notebook as nb

config, MANIFEST, REPORT_PATH = nb.make_config(MODE, MODELS, POLICIES, VIEWS, AUGMENTATION,
                                               STORAGE, MANIFEST_FILE, REPORT_RESULTS)
REPORT_ONLY = bool(REPORT_PATH)
device = None if REPORT_ONLY else nb.check_device(TARGET)

display(nb.run_summary(config, TARGET, STORAGE, SCRATCH, MANIFEST, REPORT_ONLY))
if SHOW_DETAILS or config.policies != ['none']:
    display(nb.augmentation_settings(config))
'''


def build(*, write=True):
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md(OVERVIEW),

        md(section('1. Settings',
                   'chooses where to run, how much to run, which models, and which augmentations.',
                   'nothing yet. No data or models are loaded here.',
                   '**this cell.** It is the only cell you need to change.')),
        code(settings_cell()),

        md(section('2. Setup',
                   'installs missing packages, connects persistent storage (Google Drive on Colab), '
                   'finds the project code, then checks the settings.',
                   'the folders in use and a summary of this run. Augmentation magnitudes are listed '
                   'when augmentation is on (or with `SHOW_DETAILS`).',
                   'nothing, unless your folders differ. On Colab, the shared Drive folder needs '
                   '`splits_manifest.json` and the code in `code/src/` (or set `CODE_DIR` to a repository '
                   'checkout). Restart the kernel after changing files in `src/`.')),
        code(BOOTSTRAP),
        code(CONFIGURE),

        md(section('3. Data check',
                   'verifies the dataset archive, checks every pool and episode in the shared split, '
                   'then selects this run\'s fixed episodes. Nothing is resampled.',
                   'a pass/fail summary and one development support image per class. '
                   '`SHOW_DETAILS = True` adds every table.',
                   'nothing.')),
        code('''if REPORT_ONLY:
    prepared = None
    print('Report-only mode: data check skipped.')
else:
    prepared = nb.check_data(config, STORAGE, SCRATCH, MANIFEST, show_details=SHOW_DETAILS)
'''),

        md(section('4. Run the models',
                   'for each model in turn: extracts embeddings (reusing cached ones), picks the '
                   'classifier\'s C on development episodes, then fits and scores every test episode. '
                   'Backbones stay frozen.',
                   'progress logs and the results folder. Completed work is cached, so a rerun resumes.',
                   'nothing. In report-only mode this loads the saved results instead.')),
        code('''display(nb.model_table(config))
results = nb.run_or_load(config, TARGET, STORAGE, SCRATCH, MANIFEST, prepared, REPORT_PATH)
'''),

        md(section('5. Results: model comparison',
                   'summarizes the no-augmentation results. Gains over ResNet-18 are paired on the same '
                   'episodes and given in percentage points.',
                   'a summary table and accuracy by shot. `SHOW_DETAILS = True` adds more metrics, '
                   'class recall and confusion matrices for all models.',
                   'nothing.')),
        code('nb.show_model_overview(results, show_details=SHOW_DETAILS)\n'),
        md('### Each model\n\nAccuracy, Macro-F1, the selected C, class recall and a confusion matrix at the '
           'largest shot budget. Models left out of `MODELS` are marked as not run.\n'),
        code('nb.show_each_model(results)\n'),

        md(section('6. Results: augmentation',
                   'compares each augmentation method with no augmentation on the same episodes.',
                   'the paired gain per method, the method chosen on development episodes for each '
                   'model and shot (checked on test), and per-class effects. Empty when '
                   '`POLICIES = [\'none\']`.',
                   'nothing.')),
        code('nb.show_augmentation(results, config, show_details=SHOW_DETAILS)\n'),

        md(section('7. Saved files',
                   'lists where this run\'s outputs are stored.',
                   '`results.json`, summary tables, figures (PNG and PDF) and data-check exports.',
                   'nothing.')),
        code('display(nb.output_files(results, prepared))\n'),
    ]
    for i, cell in enumerate(cells):
        cell.id = 'manifest-' + hashlib.sha256(f'{i}:{cell.source}'.encode()).hexdigest()[:12]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU', 'colab': {'gpuType': 'T4', 'provenance': []}})
    if NOTEBOOK.exists():
        old = [c for c in nbf.read(NOTEBOOK, 4).cells if c.cell_type == 'code']
        new = [c for c in notebook.cells if c.cell_type == 'code']
        if [c.source for c in old] == [c.source for c in new]:
            for before, after in zip(old, new):
                after.outputs, after.execution_count = before.outputs, before.execution_count
    nbf.validate(notebook)
    if write:
        nbf.write(notebook, NOTEBOOK)
    return notebook


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
