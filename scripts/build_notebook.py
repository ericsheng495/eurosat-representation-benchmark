"""Build the single self-contained EuroSAT research notebook."""
import argparse
import base64
import json
import ast
import hashlib
import zlib
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
MODULE_ORDER = ['config', 'cache', 'runtime', 'augment', 'data', 'models',
                'features', 'evaluation', 'pipeline', 'exploration']


def portable_source():
    blocks = []
    for name in MODULE_ORDER:
        source = (ROOT / 'src/eurosat' / f'{name}.py').read_text()
        tree = ast.parse(source)
        remove = set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.level:
                remove.update(range(node.lineno, node.end_lineno + 1))
        body = ''.join(line for i, line in enumerate(source.splitlines(True), 1) if i not in remove)
        blocks.append(f'# Source: src/eurosat/{name}.py\n' + body)
    return '\n\n'.join(blocks)


def build():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        md('''# Low-data land-use classification with frozen visual representations

**Research question.** How do pretrained CNN and Transformer representations compare on EuroSAT when only 1, 5 or 10 labeled original images per class are available, and how does support-image augmentation change that comparison?

This notebook presents the dataset audit, experimental protocol and execution workflow. Development labels select settings; held-out query labels are reserved for evaluation. Smoke runs validate execution and are not final research results.

**Colab:** open `EuroSAT_Low_Data_Benchmark.ipynb`, choose a GPU runtime (T4 or L4), then run all cells. In VS Code, select **Colab → New Colab Server → GPU → Python 3 (ipykernel)**. This single notebook embeds the project modules and works in Colab or VS Code. For a local kernel, set `TARGET = 'local'`.
'''),
        md('## 1. Runtime and execution scope'),
        code('''import os
import sys
import shutil
import subprocess
from pathlib import Path

TARGET = os.environ.get('EUROSAT_TARGET', 'colab_gpu')
MODE = os.environ.get('EUROSAT_MODE', 'smoke')  # Change to full after checking the workflow.
ASSIGNMENT = os.environ.get('EUROSAT_ASSIGNMENT', 'all')  # all or a backbone name from Parts A–D.
if TARGET not in {'colab_gpu', 'local'} or MODE not in {'smoke', 'full'}:
    raise ValueError('Choose TARGET colab_gpu/local and MODE smoke/full')
if TARGET == 'colab_gpu':
    if not Path('/content').is_dir():
        raise RuntimeError('Still using a local kernel. Select a Colab GPU server, then Run all.')
    command = shutil.which('nvidia-smi')
    gpu = subprocess.run([command, '--query-gpu=name', '--format=csv,noheader'],
                         capture_output=True, text=True) if command else None
    if gpu is None or gpu.returncode != 0 or not gpu.stdout.strip():
        raise RuntimeError('This server has no NVIDIA GPU. Create a GPU server, not a TPU/CPU server.')
    print('Colab GPU:', gpu.stdout.strip())
else:
    print('Intentional local run; no cloud GPU connection is being used.')
print('Experiment mode:', MODE)
'''),
        md('## 2. Environment and persistent storage'),
        code('''import importlib.util

requirements = {'torch': 'torch>=2.6,<3', 'torchvision': 'torchvision>=0.21,<1',
                'numpy': 'numpy>=1.26,<3', 'sklearn': 'scikit-learn>=1.4,<1.8',
                'PIL': 'Pillow>=10,<13', 'matplotlib': 'matplotlib>=3.8,<4',
                'pandas': 'pandas>=2.2,<3', 'seaborn': 'seaborn>=0.13,<0.14'}
missing = [package for module, package in requirements.items()
           if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', *missing])
if TARGET == 'colab_gpu':
    from google.colab import drive
    drive.mount('/content/drive')
    # A shared Drive folder/shortcut can be used here after granting access to all teammates.
    STORAGE = Path('/content/drive/MyDrive/ColabNotebooks/cs7643/EuroSATLowData')
    SCRATCH = Path('/content/eurosat-low-data')
else:
    current = Path.cwd()
    project = next((p for p in [current, *current.parents] if (p / 'pyproject.toml').exists()), current)
    STORAGE = Path(os.environ.get('EUROSAT_STORAGE', str(project)))
    SCRATCH = STORAGE / 'data/extracted'
STORAGE.mkdir(parents=True, exist_ok=True)
print('Persistent archive, features and results:', STORAGE)
print('Recreatable runtime-local image files:', SCRATCH)
'''),
        md('''## 3. Load project helpers

Run the next two cells once. The collapsed cell stores a compressed copy of the helpers so this single notebook works in Colab without a repository upload. The short loader makes those helpers available; experiments start later. Read or edit the implementation in `src/eurosat/`, then rebuild the notebook.
'''),
        code('''from dataclasses import asdict, replace
import json
from eurosat import Config, run
from eurosat.exploration import explore
from eurosat.runtime import check_device, environment
''', metadata={'tags': ['project_imports']}),
        md('''## 4. Experimental protocol

All ten land-use classes are retained. Decoded-RGB duplicates are checked before a deterministic, stratified 60% support-pool / 20% development / 20% held-out-test split. The 1-shot support set is nested in the corresponding 5- and 10-shot sets, with the same images and augmentation views across models.

Frozen backbones: ResNet-18, ConvNeXt-Tiny, ViT-B/16 and DINOv2 ViT-S/14. A shared multinomial logistic-regression procedure operates on L2-normalized embeddings. Development accuracy selects C and the preferred augmentation; test labels never select settings. The full preset uses 20 paired trials and original-plus-four-view conditions. See `docs/methodology.md` for strengths, checkpoint identities and interpretation limits.
'''),
        code('''config = Config.full() if MODE == 'full' else Config()
if ASSIGNMENT != 'all':
    config.models = [ASSIGNMENT]
for field in ['models', 'policies']:
    value = os.environ.get('EUROSAT_' + field.upper())
    if value:
        setattr(config, field, value.split(','))
for field in ['trials', 'views']:
    value = os.environ.get('EUROSAT_' + field.upper())
    if value:
        setattr(config, field, int(value))
config.validate()
device = check_device(TARGET)
print('Interpreter:', sys.executable)
print('Software:', environment())
print(json.dumps(asdict(config), indent=2))
print('SMOKE CHECK — not research results' if MODE == 'smoke' else 'FULL EXPERIMENT')
'''),
        md('''## 5. Dataset exploration and data-quality audit

The following analysis uses the **full EuroSAT RGB dataset**, even when the later experiment is limited to a smoke subset. pandas provides the audit tables; seaborn and matplotlib provide the figures. Cleaning removes exact-pixel duplicates before partitioning, and the split function explicitly shuffles each class with the fixed seed. These steps do not modify source images.

Example images are selected reproducibly from the support pool only. No test image is selected for the example tile or augmentation montage. Native tiles are shown without interpolation or enhancement; augmentation examples are separately identified as 224 × 224 processed views. Source: EuroSAT RGB, Helber et al. (2019), Copernicus Sentinel-2.
'''),
        code('''import pandas as pd
from IPython.display import display, Markdown, Image as NotebookImage

dataset = explore(STORAGE, SCRATCH, config)
profile = dataset['profile']
display(pd.DataFrame({
    'Measure': ['Original images', 'Retained unique images', 'Classes', 'Native image sizes',
                'Color modes', 'Duplicates removed', 'Corrupt images', 'Cross-split image overlap',
                'Cross-split pixel overlap', 'Experiment scope'],
    'Value': [profile['original_images'], profile['images'], profile['classes'],
              str(profile['native_sizes']), str(profile['color_modes']), profile['duplicates_removed'],
              profile['corrupt_images'], profile['partition_overlap_images'],
              profile['partition_overlap_pixels'], MODE]
}))
display(pd.DataFrame(profile['class_counts']).set_index('class'))
captions = [
    ('cleaning_audit', '**Figure 1. Data cleaning audit.** Archive originals versus retained images after exact-pixel duplicate removal. No duplicates are fabricated for illustration.'),
    ('class_distribution', '**Figure 2. Class balance and partitions.** Original image counts by class, divided into support, development and held-out test partitions.'),
    ('example_tile', '**Figure 3. One labeled EuroSAT image.** Native RGB tile from the annual-crop support pool. The scale marker is in pixels, not geographical distance.'),
    ('class_examples', '**Figure 4. Examples of all ten classes.** Unaugmented native RGB tiles sampled from the support pool with a fixed seed.'),
    ('stratified_shuffle', '**Figure 5. Actual stratified shuffling.** The annual-crop class is shown before and after the permutation used for the split. Dashed boundaries mark support/development/test partitions; at most 300 evenly spaced positions are plotted. Every class uses the same procedure.'),
    ('augmentation_examples', '**Figure 6. Support-image augmentation policies.** One original support image, processed at 224 × 224, shown under the baseline, repeated-original control and individual augmentation methods. These views do not add labeled original shots.')
]
for name, caption in captions:
    display(NotebookImage(filename=dataset['figures'][name]['png'], width=360 if name == 'example_tile' else 820))
    display(Markdown(caption))
print('Tables, 300-DPI PNGs, PDFs and image-source manifests:', dataset['output'])
'''),
        md('''## 6. Four-person implementation map

Choose one primary owner and one reviewer for each part. Every model owner runs the **same complete policy grid** on their own backbone; augmentation ownership below concerns implementation and review, not omitting conditions. The proposal's comparisons require shared splits, original support images, query images, strengths, classifier rules and seeds.

Set `ASSIGNMENT` in section 1 to a backbone name to run only that model. The configuration cells in Parts A–D only prepare objects; they do not start experiments. Implement reusable behavior in the linked source files, rebuild this notebook, and use a pull request. Check completed work off here and in `TODO.md`.
'''),
        md('''### Part A — Shared data protocol and ResNet-18 baseline

**Suggested owner:** teammate 1 · **Reviewer:** teammate 4

Implementation: [`data.py`](src/eurosat/data.py), [`runtime.py`](src/eurosat/runtime.py), [`models.py`](src/eurosat/models.py), [`exploration.py`](src/eurosat/exploration.py).

- [x] Verified RGB archive, decoded-pixel duplicate checks and class labels.
- [x] Stratified 60/20/20 partitions and nested, paired 1/5/10-shot support sets.
- [x] Frozen ResNet-18 adapter and full-dataset EDA figures.
- [ ] Agree on and freeze the shared dataset/seed/protocol manifest.
- [ ] Complete the full ResNet grid and export `results.json`.
- [ ] Have a second teammate verify data isolation and reproducibility.
- [ ] Resolve geographic-grouping feasibility or document the limitation.
'''),
        code('''# Part A implementation TODO: make shared-protocol changes in data.py/config.py,
# and the ResNet representation adapter in models.py. Do not change test partitions here.
# Set ASSIGNMENT = 'resnet18' in section 1 for this teammate's execution.
part_a_config = replace(config, models=['resnet18'])
print('Part A:', part_a_config.models)
'''),
        md('''### Part B — ConvNeXt-Tiny and geometric augmentation validation

**Suggested owner:** teammate 2 · **Reviewer:** teammate 1

Implementation: [`models.py`](src/eurosat/models.py), [`augment.py`](src/eurosat/augment.py), [`features.py`](src/eurosat/features.py).

- [x] ImageNet-1K ConvNeXt-Tiny, pooled features and final checkpoint normalization.
- [x] Flip, rotation, translation and scaling policies with deterministic image/view seeds.
- [ ] Review geometric strengths using development data; freeze choices before test reporting.
- [ ] Complete the full ConvNeXt grid, including every non-geometric control/condition.
- [ ] Report paired gains and compare modern-CNN versus ResNet representations.
- [ ] Review the shared split and augmentation-view manifests with Part A.
'''),
        code('''# Part B implementation TODO: geometric transforms belong in augment.py;
# ConvNeXt feature extraction belongs in models.py. Reuse the common classifier.
# Set ASSIGNMENT = 'convnext_tiny' in section 1 to execute this model only.
part_b_config = replace(config, models=['convnext_tiny'])
print('Part B:', part_b_config.models)
'''),
        md('''### Part C — ViT-B/16 and color/pixel augmentation validation

**Suggested owner:** teammate 3 · **Reviewer:** teammate 2

Implementation: [`models.py`](src/eurosat/models.py), [`augment.py`](src/eurosat/augment.py), [`features.py`](src/eurosat/features.py).

- [x] ImageNet-1K ViT-B/16 final normalized class-token representation.
- [x] Brightness/contrast, noise, blur and erasing conditions, with support-only augmentation.
- [ ] Confirm preprocessing, feature dimensions and frozen evaluation behavior.
- [ ] Review policy strengths on development data; keep views identical across models.
- [ ] Complete the full ViT grid and export paired comparisons with both CNNs.
- [ ] Prespecify selected 1-shot nine-view comparisons; cutout remains optional.
'''),
        code('''# Part C implementation TODO: color/noise/blur/erasing policies belong in augment.py;
# ViT representation extraction belongs in models.py. Keep original shot budgets unchanged.
# Set ASSIGNMENT = 'vit_b_16' in section 1 to execute this model only.
part_c_config = replace(config, models=['vit_b_16'])
print('Part C:', part_c_config.models)
'''),
        md('''### Part D — DINOv2, statistical summaries and report integration

**Suggested owner:** teammate 4 · **Reviewer:** teammate 3

Implementation: [`models.py`](src/eurosat/models.py), [`evaluation.py`](src/eurosat/evaluation.py), [`reporting.py`](src/eurosat/reporting.py).

- [x] Official pinned DINOv2 ViT-S/14 checkpoint and frozen class-token features.
- [x] Common logistic regression, Macro-F1, recall, confusion matrices and paired gains.
- [ ] Complete the full DINOv2 grid and validate checkpoint/software provenance.
- [ ] Merge compatible model exports and check trial completeness/paired comparisons.
- [ ] Prepare model-by-shot, augmentation-gain and class-level result figures.
- [ ] Report development-label use, timing conditions, negative findings and pretraining confounds.
- [ ] Coordinate final methods, limitations and citations; do not attribute DINOv2/ViT differences to self-supervision alone.
'''),
        code('''# Part D implementation TODO: keep metric/classifier rules in evaluation.py;
# combine completed exports through reporting.py rather than hand-editing CSVs.
# Set ASSIGNMENT = 'dinov2_vits14' in section 1 to execute this model only.
part_d_config = replace(config, models=['dinov2_vits14'])
print('Part D:', part_d_config.models)
'''),
        md('''## 7. Run or resume the selected experiment

Only support images receive random augmentation. Development and test preprocessing are deterministic. Feature extraction is cached in chunks, classifier parameters are stored as numeric checkpoints, and completed trials are skipped. This cell does not repeat dataset cleaning or GPU work that has already completed for the same configuration.
'''),
        code('results = run(config, STORAGE, SCRATCH, target=TARGET)\n'),
        md('''## 8. Results and reporting

Accuracy and Macro-F1 are fractions; paired improvements are percentage points relative to each trial's unaugmented baseline. Preferred policies are selected from development performance, while all prespecified comparisons remain available in the exported results. Smoke scores below validate execution only.
'''),
        code('''table = pd.DataFrame(results['summary'])
display(table.sort_values(['model', 'shots', 'policy']).reset_index(drop=True))
print('Results, classifier records and development-selected policies:', results['output'])
'''),
    ]
    # Keep measured dataset previews visible before connecting a runtime.
    # Running the EDA cell replaces these saved outputs with the current audit.
    audit = next(c for c in notebook.cells if c.cell_type == 'code' and
                 'dataset = explore(STORAGE, SCRATCH, config)' in c.source)
    figures = ROOT / 'docs/figures'
    profile = json.loads((figures / 'dataset_profile.json').read_text())
    audit.outputs = [nbf.v4.new_output('display_data', data={'text/markdown':
        '**Saved dataset preview** — measured from the verified EuroSAT archive. '
        'Run this cell to refresh the audit for your storage and configuration.\n\n'
        f"**{profile['images']:,} images · {profile['classes']} classes · "
        f"{profile['duplicates_removed']} exact duplicates removed.**"})]
    previews = [('cleaning_audit', 'Cleaning: original and retained image counts'),
                ('class_distribution', 'Class balance and support/development/test partitions'),
                ('example_tile', 'One native RGB support image: Annual crop'),
                ('class_examples', 'Native support images from all ten classes'),
                ('stratified_shuffle', 'Actual stratified shuffling before partitioning'),
                ('augmentation_examples', 'Processed support-image augmentation examples')]
    for name, caption in previews:
        audit.outputs.append(nbf.v4.new_output('display_data', data={
            'text/markdown': '**' + caption + '**'}))
        audit.outputs.append(nbf.v4.new_output('display_data', data={
            'image/png': base64.b64encode((figures / (name + '.png')).read_bytes()).decode(),
            'text/plain': caption}, metadata={'image/png': {
                'width': 360 if name == 'example_tile' else 820}}))
    notebook.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python'}, 'accelerator': 'GPU',
                         'colab': {'gpuType': 'T4', 'provenance': []}}
    index = next(i for i, c in enumerate(notebook.cells)
                 if 'project_imports' in c.metadata.get('tags', []))
    payload = base64.b64encode(zlib.compress(portable_source().encode(), level=9)).decode()
    packed = code('# Generated project payload — edit src/eurosat/, then rebuild.\n'
                  + '_EUROSAT_SOURCE = ' + repr(payload), metadata={
                      'tags': ['generated_project_payload'],
                      'jupyter': {'source_hidden': True}, 'collapsed': True,
                      'colab': {'source_hidden': True}})
    loader = code('''# Load the bundled helpers; no experiment runs in this cell.
import base64
import zlib
exec(compile(zlib.decompress(base64.b64decode(_EUROSAT_SOURCE)),
             'eurosat-bundled-helpers', 'exec'), globals())
print('EuroSAT helpers ready. Continue to configuration and dataset exploration.')
''', metadata={'tags': ['project_loader']})
    notebook.cells[index:index + 1] = [packed, loader]
    for i, cell in enumerate(notebook.cells):
        cell.id = 'cell-' + hashlib.sha256(f'{i}:{cell.source}'.encode()).hexdigest()[:12]
    destination = ROOT / 'EuroSAT_Low_Data_Benchmark.ipynb'
    nbf.validate(notebook)
    nbf.write(notebook, destination)
    return destination



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Fail if the tracked notebook has drifted')
    args = parser.parse_args()
    path = ROOT / 'EuroSAT_Low_Data_Benchmark.ipynb'
    before = nbf.read(path, as_version=4) if args.check and path.exists() else None
    outputs = build()
    if args.check:
        after = nbf.read(path, as_version=4)
        # Generated cell IDs are incidental; source, type and metadata must agree.
        canonical = lambda nb: [(c.cell_type, c.source, dict(c.metadata)) for c in nb.cells]
        if before is None or canonical(before) != canonical(after):
            raise SystemExit('Tracked notebook is stale; run python scripts/build_notebook.py and commit it.')
    print(outputs)
