"""Running the benchmark and reading its results.

  conditions   what the classifier is trained on (tracks 1-3 and the view sweep)
  embeddings   frozen-backbone embeddings, cached per backbone, image and view
  classifier   C chosen on development episodes, then every test episode scored
  runner          every baseline x model x condition x shot, resumable
  feature_check   many-label check per backbone (table T5)
  results         per-episode metrics, summaries and paired comparisons
"""
from .conditions import Condition, build_conditions
from .results import Results, load
from .runner import run

__all__ = ['Condition', 'Results', 'build_conditions', 'load', 'run']
