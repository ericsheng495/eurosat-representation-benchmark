"""Generate a readable, self-contained notebook; upload this one file to Colab."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
notebook = nbf.v4.new_notebook()
notebook.cells = [
    md('''# EuroSAT: Low-Data Land-Use Classification

A fresh experiment based on the project proposal: frozen backbones, ten classes, nested 1/5/10-shot support sets, and one L2-regularized multinomial logistic-regression classifier.

**Start here:** upload this notebook to Google Colab, choose **Runtime → Change runtime type → T4 GPU** (or another GPU), connect, then **Run all**. Authorize Drive when prompted. In VS Code, choose **Select Another Kernel → Colab → New Colab Server → GPU → Python 3 (ipykernel)**.

The default **smoke** run checks the workflow with real images and ResNet-18. Its scores are not research results. After it succeeds, set `MODE = "full"` below to run the configured four-model experiment. A TPU server will not run this GPU notebook.
'''),
    md('## 1. Connection and run settings'),
    code('''import os
import sys
import shutil
import subprocess
from pathlib import Path

TARGET = os.environ.get('EUROSAT_TARGET', 'colab_gpu')  # Keep colab_gpu for cloud execution.
MODE = os.environ.get('EUROSAT_MODE', 'smoke')          # smoke first; full for the experiment.
if TARGET not in {'colab_gpu', 'local'} or MODE not in {'smoke', 'full'}:
    raise ValueError('Choose TARGET colab_gpu/local and MODE smoke/full')
if TARGET == 'colab_gpu':
    if not Path('/content').is_dir():
        raise RuntimeError('This is still a local kernel. Select a Colab GPU server, then Run all.')
    command = shutil.which('nvidia-smi')
    gpu = (subprocess.run([command, '--query-gpu=name', '--format=csv,noheader'],
                          capture_output=True, text=True) if command else None)
    if gpu is None or gpu.returncode != 0 or not gpu.stdout.strip():
        raise RuntimeError('This server has no NVIDIA GPU. Create a GPU server (T4/L4), '
                           'not a TPU or CPU server, then select its Python kernel.')
    print('Colab GPU server:', gpu.stdout.strip())
else:
    print('Intentional local check. Cloud connection is not being used.')
print('Run mode:', MODE)
'''),
    md('## 2. Dependencies and persistent storage'),
    code('''import importlib.util

requirements = {'torch': 'torch>=2.6,<3', 'torchvision': 'torchvision>=0.21,<1',
                'numpy': 'numpy>=1.26,<3', 'sklearn': 'scikit-learn>=1.4,<1.8',
                'PIL': 'Pillow>=10,<13'}
missing = [package for module, package in requirements.items()
           if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', *missing])

if TARGET == 'colab_gpu':
    from google.colab import drive
    drive.mount('/content/drive')
    STORAGE = Path('/content/drive/MyDrive/ColabNotebooks/cs7643/EuroSATLowData')
    SCRATCH = Path('/content/eurosat-low-data')
else:
    STORAGE = Path(os.environ.get('EUROSAT_STORAGE', str(Path.cwd())))
    SCRATCH = STORAGE / 'data/extracted'
STORAGE.mkdir(parents=True, exist_ok=True)
print('Persistent files:', STORAGE)
print('Extracted images (can be recreated):', SCRATCH)
'''),
    md('''## 3. Load pipeline helpers

Run this cell once per session. The helpers are embedded so you only need to upload this notebook. Original data, feature chunks, classifier checkpoints and finished trials are cached separately.'''),
    code((root / 'src/eurosat/pipeline.py').read_text(),
         metadata={'jupyter': {'source_hidden': True}}),
    md('''## 4. Confirm hardware and experiment configuration

Full mode: ResNet-18, ConvNeXt-Tiny, ViT-B/16 and official DINOv2 ViT-S/14; 20 paired trials; 1/5/10 shots; original + four augmented views; separate augmentation policies and repeated-original controls. All backbones stay frozen. Development labels select classifier C and the preferred augmentation; held-out test labels do not select settings.

You may reduce `config.models`, `config.policies` or `config.trials` for a smaller prespecified experiment. Keep `none` in policies for paired baseline comparisons. Check the printed configuration before proceeding.'''),
    code('''config = Config.full() if MODE == 'full' else Config()
if os.environ.get('EUROSAT_MODELS'):
    config.models = os.environ['EUROSAT_MODELS'].split(',')
if os.environ.get('EUROSAT_POLICIES'):
    config.policies = os.environ['EUROSAT_POLICIES'].split(',')
if os.environ.get('EUROSAT_TRIALS'):
    config.trials = int(os.environ['EUROSAT_TRIALS'])
if os.environ.get('EUROSAT_VIEWS'):
    config.views = int(os.environ['EUROSAT_VIEWS'])
config.validate()
device = check_device(TARGET)
print('Python:', sys.executable)
print('Software:', environment())
print(json.dumps(asdict(config), indent=2))
print('SMOKE CHECK — not benchmark results' if MODE == 'smoke' else 'FULL EXPERIMENT')
'''),
    md('''## 5. Run or resume

The RGB archive downloads once. Images are extracted to fast runtime-local disk, exact-pixel duplicates are checked before the 60/20/20 split, and no class folders are copied into multiple Drive directories. Each completed feature chunk and classifier trial is saved. Rerunning resumes missing work; completed model results skip model loading entirely.

Full mode is a substantial grid. Interrupting it preserves completed chunks/trials. The original support images are identical across models and augmentation policies, and 1-shot sets are nested in 5- and 10-shot sets.'''),
    code('results = run(config, STORAGE, SCRATCH, target=TARGET)\n'),
    md('## 6. Inspect saved results'),
    code('''from IPython.display import HTML, display
from html import escape

columns = ['model', 'policy', 'shots', 'trials', 'accuracy', 'macro_f1', 'paired_gain_percentage_points']
header = '<tr>' + ''.join('<th>' + escape(name) + '</th>' for name in columns) + '</tr>'
body = ''
for row in results['summary']:
    values = [f'{row[name]:.4f}' if isinstance(row[name], float) else str(row[name]) for name in columns]
    body += '<tr>' + ''.join('<td>' + escape(value) + '</td>' for value in values) + '</tr>'
display(HTML('<table>' + header + body + '</table>'))
print('Saved JSON, CSV and development-selected policies:', results['output'])
'''),
]
notebook.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                     'language_info': {'name': 'python'}, 'accelerator': 'GPU',
                     'colab': {'gpuType': 'T4', 'provenance': []}}
nbf.validate(notebook)
nbf.write(notebook, root / 'Start_Here.ipynb')
print(root / 'Start_Here.ipynb')
