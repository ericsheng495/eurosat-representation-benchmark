"""Figures and tables for the report (figure_n_table_sug.md), built from a run's results.

  style     shared fonts, colors and drawing helpers
  figures   F1-F11
  tables    T1-T3
  report    builds all 16 items and saves them next to the results
"""
from .style import STYLE, save_figure


def build_report(*args, **kwargs):
    from .report import build_report as build  # Imported on use: it needs the models and results code.
    return build(*args, **kwargs)


__all__ = ['STYLE', 'save_figure', 'build_report']
