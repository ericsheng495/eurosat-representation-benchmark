"""Verify the actual portable notebook in an isolated local Jupyter kernel."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['full', 'smoke'], help='Default: full benchmark, smoke fine-tuning')
    parser.add_argument('--models', help='Default: DINOv2')
    parser.add_argument('--policies')
    parser.add_argument('--trials', type=int)
    parser.add_argument('--views', type=int)
    parser.add_argument('--experiment', choices=['primary', 'part-d'])
    parser.add_argument('--notebook', type=Path, help='Defaults to the generated self-contained Colab notebook')
    parser.add_argument('--output', type=Path, help='Save outputs here; defaults to the input notebook')
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.notebook or root / 'EuroSAT_Representation_Benchmark.ipynb'
    is_finetuning = source.name == 'EuroSAT_Fine_Tuning.ipynb'
    mode = args.mode or ('smoke' if is_finetuning else 'full')
    models = args.models or 'dinov2_vits14'
    if is_finetuning and models != 'dinov2_vits14':
        raise ValueError('The partial fine-tuning notebook focuses on dinov2_vits14')
    output = args.output or source
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(source, as_version=4)
    started = time.monotonic()
    def on_cell_start(cell, cell_index, **kwargs):
        if cell.cell_type == 'code':
            print(f'Cell {cell_index}: running', flush=True)
    def on_cell_executed(cell, **kwargs):
        for block in cell.get('outputs', []):
            if block.output_type == 'stream':
                print(block.text, end='', flush=True)
    with tempfile.TemporaryDirectory(prefix='eurosat-representation-benchmark-kernel-') as temporary:
        spec = Path(temporary) / 'eurosat-fresh'
        spec.mkdir()
        (spec / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'EuroSAT fresh environment', 'language': 'python'}))
        manager = KernelManager(kernel_name='eurosat-fresh',
                                client_class='jupyter_client.asynchronous.AsyncKernelClient',
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]))
        client = NotebookClient(notebook, km=manager, timeout=7200, iopub_timeout=120,
                                on_cell_start=on_cell_start, on_cell_executed=on_cell_executed)
        env = dict(os.environ, EUROSAT_TARGET='local', EUROSAT_MODE=mode,
                   EUROSAT_STORAGE=str(root), EUROSAT_MODELS=models,
                   EUROSAT_FINETUNE_MODE=mode, EUROSAT_FINETUNE_MODEL=models,
                   EUROSAT_EXPERIMENT=args.experiment or ('part-d' if models == 'dinov2_vits14' else 'primary'),
                   MPLBACKEND='Agg')
        for field in ['policies', 'trials', 'views']:
            value = getattr(args, field)
            if value is not None:
                env['EUROSAT_' + field.upper()] = str(value)
        try:
            client.execute(cwd=str(root), env=env)
        finally:
            nbformat.write(notebook, output)
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
                manager.cleanup_resources()
    print(f'Notebook passed in {time.monotonic() - started:.1f}s: {output}', flush=True)


if __name__ == '__main__':
    main()
