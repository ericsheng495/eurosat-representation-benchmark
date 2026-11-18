"""Build every figure and table of the report from one run, and save them next to its results.

<run>/report/
  F1_augmented_views.png/.pdf ... F11_baseline_effects.png/.pdf
  T1_main_results.csv ... T4_labels_used.csv (each + _numbers.csv)
  report.json   which items were made, where, and why any were skipped
"""
from dataclasses import dataclass, field
from pathlib import Path

from ..cache import write_json
from ..config import Config
from ..training import finetuned_weights_dir
from . import figures as F
from . import tables as T

# (id, title, placement in the report) in the order of figure_n_table_sug.md.
ITEMS = [('F1', 'Augmented support views', 'supplementary'), ('T2', 'Models and fine-tuning', 'supplementary'),
         ('T4', 'Labels used', 'supplementary'), ('T1', 'Main results', 'main'), ('F2', 'Few-shot curves', 'main'),
         ('F3', 'Paired model differences', 'main'), ('F4', 'Per-class recall', 'supplementary'),
         ('F5', 'Confusion matrices', 'supplementary'), ('F11', 'Baseline effects', 'main'),
         ('F6', 'Augmentation gains', 'main'), ('F7', 'All-augmentation gain by shot', 'supplementary'),
         ('F8', 'Label-gap closure', 'main'), ('T3', 'Preferred augmentation', 'main'),
         ('F9', 'Number of views', 'supplementary'), ('F10', 'Class-level augmentation effects', 'supplementary')]


@dataclass
class Report:
    output: Path
    figures: dict = field(default_factory=dict)   # id -> PNG path
    tables: dict = field(default_factory=dict)    # id -> table to read
    skipped: dict = field(default_factory=dict)   # id -> reason

    def placement(self, item):
        return next(place for i, _, place in ITEMS if i == item)

    def title(self, item):
        return next(title for i, title, _ in ITEMS if i == item)


def build_report(results, prepared=None, config=None, output=None):
    """Make every item in ITEMS. Items the run cannot support are skipped with a reason, never faked."""
    config = config or Config.from_dict(results.meta['config'])
    output = Path(output or results.path.parent / 'report')
    output.mkdir(parents=True, exist_ok=True)
    report = Report(output)
    weights_dir = finetuned_weights_dir(results.storage, smoke=config.mode == 'smoke')
    makers = {
        'F1': lambda: F.f1_augmented_views(prepared, config, output),
        'F2': lambda: F.f2_fewshot_curves(results, output), 'F3': lambda: F.f3_model_differences(results, output),
        'F4': lambda: F.f4_class_recall(results, output), 'F5': lambda: F.f5_confusion(results, output),
        'F6': lambda: F.f6_augmentation_gains(results, output), 'F7': lambda: F.f7_all_gain_by_shot(results, output),
        'F8': lambda: F.f8_label_gap(results, output), 'F9': lambda: F.f9_views(results, output),
        'F10': lambda: F.f10_class_effects(results, output), 'F11': lambda: F.f11_baseline_effects(results, output),
        'T1': lambda: T.t1_main_results(results), 'T2': lambda: T.t2_models(results, weights_dir),
        'T3': lambda: T.t3_preferred_augmentation(results), 'T4': lambda: T.t4_labels_used(results, weights_dir),
    }
    names = {'T1': 'T1_main_results', 'T2': 'T2_models', 'T3': 'T3_preferred_augmentation', 'T4': 'T4_labels_used'}
    for item, _, _ in ITEMS:
        try:
            made = makers[item]()
        except F.MissingData as reason:
            report.skipped[item] = str(reason)
            continue
        if item.startswith('F'):
            report.figures[item] = made
        else:
            table, numbers = made
            table.to_csv(output / f'{names[item]}.csv')
            numbers.to_csv(output / f'{names[item]}_numbers.csv', index=False)
            report.tables[item] = table
    write_json(output / 'report.json', {
        'results': str(results.path), 'items': [
            {'id': item, 'title': title, 'placement': place,
             'files': ([f'{report.figures[item].stem}.png', f'{report.figures[item].stem}.pdf'] if item in report.figures
                       else [f'{names[item]}.csv'] if item in report.tables else []),
             'skipped': report.skipped.get(item)} for item, title, place in ITEMS]})
    made = len(report.figures) + len(report.tables)
    print(f'Report: {made} of {len(ITEMS)} items saved to {output}'
          + (f' (skipped: {", ".join(report.skipped)})' if report.skipped else ''), flush=True)
    return report
