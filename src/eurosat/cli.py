"""Command line: the same steps as the notebook, for long runs outside Jupyter.

  python -m eurosat finetune --models all          fine-tune for the fine-tuned baseline
  python -m eurosat run --mode full                 run or resume the benchmark
"""
import argparse
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog='eurosat')
    commands = parser.add_subparsers(dest='command', required=True)
    storage = Path(os.environ.get('EUROSAT_STORAGE', str(Path.cwd())))

    experiment = commands.add_parser('run', help='Run or resume the benchmark')
    experiment.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    experiment.add_argument('--models', nargs='+', default=['all'])
    experiment.add_argument('--baselines', nargs='+', default=['all'])
    experiment.add_argument('--tracks', nargs='+', type=int, default=[1, 2, 3])
    experiment.add_argument('--no-view-sweep', action='store_true')
    experiment.add_argument('--manifest', type=Path)
    experiment.add_argument('--storage', type=Path, default=storage)

    finetune = commands.add_parser('finetune', help='Fine-tune models on the six base classes (fine-tuned baseline)')
    finetune.add_argument('--models', nargs='+', default=['all'])
    finetune.add_argument('--manifest', type=Path)
    finetune.add_argument('--storage', type=Path, default=storage)
    finetune.add_argument('--workers', type=int, default=4, help='image-loading processes')
    finetune.add_argument('--smoke', action='store_true', help='2 batches per epoch into a separate folder, to test the code')
    finetune.add_argument('--overwrite', action='store_true', help='retrain even if weights already exist')
    args = parser.parse_args(argv)

    from .config import BASELINES, MODEL_IDS, Config
    models = list(MODEL_IDS) if args.models == ['all'] else args.models
    manifest = args.manifest or args.storage / 'splits_manifest.json'
    scratch = args.storage / 'data/extracted'
    if args.command == 'finetune':
        from .training import fine_tune
        return fine_tune(models, args.storage, scratch, manifest, workers=args.workers, smoke=args.smoke,
                         overwrite=args.overwrite)

    from .experiments import run
    from .preprocessing.manifest import prepare_manifest_data
    baselines = list(BASELINES) if args.baselines == ['all'] else args.baselines
    config = Config.for_mode(args.mode, models=models, baselines=baselines, tracks=args.tracks,
                             view_sweep=not args.no_view_sweep)
    prepared = prepare_manifest_data(args.storage, scratch, config, manifest)
    return run(config, args.storage, scratch, manifest, prepared, target='local')
