"""Command-line entry points for experiments and full-dataset exploration."""
import argparse
from pathlib import Path

from .config import Config
from .pipeline import run


def main(argv=None):
    parser = argparse.ArgumentParser(prog='eurosat', description='Few-shot EuroSAT research workflow')
    commands = parser.add_subparsers(dest='command', required=True)
    experiment = commands.add_parser('run', help='Run or resume an experiment locally')
    experiment.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    experiment.add_argument('--models', nargs='+')
    experiment.add_argument('--policies', nargs='+')
    experiment.add_argument('--trials', type=int)
    experiment.add_argument('--views', type=int)
    experiment.add_argument('--storage', type=Path, default=Path.cwd())
    exploration = commands.add_parser('explore', help='Audit the full dataset and export paper figures')
    exploration.add_argument('--storage', type=Path, default=Path.cwd())
    exploration.add_argument('--output', type=Path)
    merge = commands.add_parser('merge', help='Combine compatible completed teammate result exports')
    merge.add_argument('inputs', nargs='+', type=Path)
    merge.add_argument('--output', required=True, type=Path)
    report = commands.add_parser('report', help='Render figures and tables from completed results')
    report.add_argument('results', type=Path)
    report.add_argument('--output', type=Path)
    suite = commands.add_parser('part-d', help='Run the proposal-aligned DINOv2 experiment suite')
    suite.add_argument('--mode', choices=['smoke', 'full'], default='full')
    suite.add_argument('--storage', type=Path, default=Path.cwd())
    suite.add_argument('--skip-strength-selection', action='store_true')
    suite.add_argument('--include-cutout', action='store_true')
    suite.add_argument('--skip-resnet-baseline', action='store_true')
    suite.add_argument('--selected-methods', type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == 'part-d':
        from .experiments import run_part_d
        return run_part_d(args.storage, args.storage / 'data/extracted', target='local',
                          mode=args.mode, select_strengths=not args.skip_strength_selection,
                          include_cutout=args.include_cutout, selected_methods=args.selected_methods,
                          include_resnet=not args.skip_resnet_baseline)
    if args.command == 'report':
        import matplotlib
        matplotlib.use('Agg')
        from .reporting import create_report
        return create_report(args.results, args.output)
    if args.command == 'merge':
        from .reporting import merge_results
        return merge_results(args.inputs, args.output)
    scratch = args.storage / 'data/extracted'
    if args.command == 'explore':
        import matplotlib
        matplotlib.use('Agg')
        from .exploration import explore
        return explore(args.storage, scratch, output=args.output)
    config = Config.full() if args.mode == 'full' else Config()
    for name in ['models', 'policies', 'trials', 'views']:
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    return run(config, args.storage, scratch, target='local')
