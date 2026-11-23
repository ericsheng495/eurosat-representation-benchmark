"""EuroSAT few-shot benchmark.

  config          settings, model checkpoints and augmentation defaults
  preprocessing/  dataset checks, fixed episodes and image transforms
  models/         four models, three baselines each
  training/       fine-tuning for the fine-tuned baseline
  experiments/    running the benchmark and reading its results
  reporting/      figures and tables
  notebook        one helper per notebook section
"""
from .config import Config

__all__ = ['Config']
