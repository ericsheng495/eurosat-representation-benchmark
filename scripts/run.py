"""Run the pipeline locally; cloud notebook execution starts at Start_Here.ipynb."""
import argparse
from pathlib import Path
from eurosat.pipeline import Config, run

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
parser.add_argument('--models', nargs='+', help='Override selected backbones')
parser.add_argument('--policies', nargs='+', help='Override augmentation conditions (include none)')
parser.add_argument('--trials', type=int)
parser.add_argument('--views', type=int)
args = parser.parse_args()
config = Config.full() if args.mode == 'full' else Config()
if args.models:
    config.models = args.models
if args.policies:
    config.policies = args.policies
if args.trials is not None:
    config.trials = args.trials
if args.views is not None:
    config.views = args.views
root = Path(__file__).resolve().parents[1]
run(config, root, root / 'data/extracted', target='local')
