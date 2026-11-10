"""Run fixed four-class episodes, or render existing model results."""
import argparse
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog='eurosat')
    commands = parser.add_subparsers(dest='command', required=True)
    experiment = commands.add_parser('run', help='Run or resume the manifest benchmark')
    experiment.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    experiment.add_argument('--models', nargs='+')
    experiment.add_argument('--manifest', type=Path)
    experiment.add_argument('--policies', nargs='+')
    experiment.add_argument('--trials', type=int)
    experiment.add_argument('--views', type=int)
    experiment.add_argument('--storage', type=Path, default=Path(os.environ.get('EUROSAT_STORAGE', str(Path.cwd()))))
    report = commands.add_parser('report', help='Render an existing results.json without model inference')
    report.add_argument('results', type=Path)
    report.add_argument('--output', type=Path)
    args = parser.parse_args(argv)

    if args.command == 'report':
        from .reporting import create_report
        return create_report(args.results, args.output)

    from .experiments import manifest_config, run_manifest_benchmark
    config = manifest_config(args.mode, args.models or os.environ.get('EUROSAT_MODELS', 'all'))
    for name in ['policies', 'views']:
        if getattr(args, name) is not None:
            setattr(config, name, getattr(args, name))
    return run_manifest_benchmark(config, args.storage, args.storage / 'data/extracted',
                                  args.manifest or args.storage / 'splits_manifest.json', target='local',
                                  episode_limit=args.trials)
