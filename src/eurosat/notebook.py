"""Helpers that keep the benchmark notebook short: one call per notebook section.

The notebook shows *what* runs; this module holds *how*. Every helper uses the
same shared functions as the command line, so results do not depend on the entry point.
"""
import copy
import json
import os
from pathlib import Path

from .config import MODEL_IDS, Config, augmentation_table
from .runtime import check_device, environment

MODEL_INFO = {
    'resnet18': ('ResNet-18', 'Standard CNN (baseline)'),
    'convnext_tiny': ('ConvNeXt-Tiny', 'Modern CNN'),
    'vit_b_16': ('ViT-B/16', 'Supervised Transformer'),
    'dinov2_vits14': ('DINOv2 ViT-S/14', 'Self-supervised Transformer'),
}

__all__ = ['check_device', 'make_config', 'run_summary', 'augmentation_settings', 'check_data',
           'model_table', 'run_or_load', 'show_model_overview', 'show_each_model',
           'show_augmentation', 'output_files']


def _show(*items):
    from IPython.display import display
    display(*items)


def _heading(text):
    from IPython.display import Markdown
    _show(Markdown(text))


def _figure(path, width=900):
    from IPython.display import Image
    _show(Image(filename=str(path), width=width))


def _override(name, value, variable, convert=str):
    """Automated runs (scripts/execute_notebook.py) pass settings as EUROSAT_* variables."""
    raw = os.environ.get(variable)
    if not raw:
        return value
    print(f'{name} = {raw!r}  (from environment variable {variable})', flush=True)
    return convert(raw)


def make_config(mode, models, policies, views, augmentation, storage, manifest_file='', report_results=''):
    """Turn the Settings cell into a validated Config. Returns (config, manifest path, report path)."""
    mode = _override('MODE', mode, 'EUROSAT_MODE')
    models = _override('MODELS', models, 'EUROSAT_MODELS')
    policies = _override('POLICIES', policies, 'EUROSAT_POLICIES', lambda raw: raw.split(','))
    views = _override('VIEWS', views, 'EUROSAT_VIEWS', int)
    manifest_file = _override('MANIFEST_FILE', manifest_file, 'EUROSAT_SPLITS')
    report_results = _override('REPORT_RESULTS', report_results, 'EUROSAT_REPORT_RESULTS')

    if report_results:
        saved = json.loads(Path(report_results).read_text())
        if 'episode_source' not in saved['manifest']:
            raise ValueError('REPORT_RESULTS must point to results from the four-class manifest benchmark')
        config = Config.from_dict(saved['manifest']['config'])
    else:
        from .experiments import manifest_config
        config = manifest_config(mode, models)
        config.policies, config.views = list(policies), int(views)
        config.augmentation = copy.deepcopy(augmentation)
        config.trials = _override('test episodes', config.trials, 'EUROSAT_TRIALS', int)
    config.validate()
    manifest = Path(manifest_file) if manifest_file else Path(storage) / 'splits_manifest.json'
    return config, manifest, str(report_results)


def run_summary(config, target, storage, scratch, manifest, report_only=False):
    """One table with everything this run will use."""
    import pandas as pd
    dev = 2 if config.mode == 'smoke' else 100
    methods = [p for p in config.policies if p != 'none']
    rows = [
        ('Mode', f"{config.mode} ({'quick execution check' if config.mode == 'smoke' else 'research run'})"),
        ('Episodes per shot', f'{dev} development + {config.trials} test'),
        ('Shots per class', ', '.join(map(str, config.shots))),
        ('Models', ', '.join(MODEL_INFO[m][0] for m in config.models)),
        ('Augmentation', f"{', '.join(methods)} · {config.views} views per support image" if methods else 'none'),
        ('Target', target),
        ('Shared storage', str(storage)),
        ('Local images', str(scratch)),
        ('Split manifest', str(manifest)),
        ('Software', ', '.join(f'{k} {v}' for k, v in environment().items())),
    ]
    if report_only:
        rows.insert(0, ('Report only', 'redraw saved results; no data download or models'))
    return pd.DataFrame(rows, columns=['setting', 'value']).set_index('setting')


def augmentation_settings(config):
    """Every augmentation magnitude next to its default."""
    import pandas as pd
    table = pd.DataFrame(augmentation_table(config.augmentation))
    table.insert(1, 'in_run', table.policy.isin(config.policies))
    no_magnitude = table.parameter == '-'  # flip and rotation have no strength setting.
    table[['value', 'default']] = table[['value', 'default']].astype(object)
    table.loc[no_magnitude, ['value', 'default']] = '—'
    return table


def check_data(config, storage, scratch, manifest, show_details=False):
    """Verify the archive, the full manifest and this run's episodes; show a pass/fail summary."""
    import pandas as pd
    from .preprocessing import load_source, validate_manifest, select_episodes
    from .preprocessing.audit import episode_table, usage_table, inspect_batch, data_figures

    source = load_source(storage, scratch, config.seed)
    validated = validate_manifest(manifest, source)
    prepared = select_episodes(source, validated, config, storage)
    batch = inspect_batch(prepared, config)
    usage = usage_table(prepared)  # Also exported as episode-usage.csv.
    figures = data_figures(prepared, validated)

    integrity = source.integrity
    dev = len(next(iter(prepared.episodes['dev'].values())))
    rows = [
        ('Archive checksum (MD5)', 'passed', 'EuroSAT.zip matches the published checksum'),
        ('Extracted files (CRC)', 'passed', f"{integrity['files_crc_verified']:,} files; "
                                            f"{integrity['missing_or_changed_files_restored']} restored"),
        ('Image decoding and duplicates', 'passed', f"{integrity['corrupt_images']} corrupt; "
                                                    f"{integrity['decoded_pixel_duplicates']} exact-pixel duplicates"),
        *[(check['check'], check['result'], 'every pool and episode in the manifest') for check in validated.checks],
        ('Episodes selected', 'passed', f'{dev} development + {config.trials} test per shot; '
                                        f'{len(prepared.records):,} unique images'),
        ('Model input', 'passed', f"{tuple(batch['batch_shape'][1:])} tensors; finite; identical on re-read"),
    ]
    _heading('**Data check summary**')
    _show(pd.DataFrame(rows, columns=['check', 'result', 'detail']).set_index('check'))
    _figure(figures['support_examples'], 700)

    if show_details:
        _heading('**Source inventory** (all ten classes; six are reserved)')
        _show(pd.DataFrame(source.inventory))
        print('Native image properties:', source.properties)
        _heading('**Manifest pools and episode counts**')
        _show(pd.DataFrame(validated.pools), pd.DataFrame(validated.episode_counts))
        _figure(figures['novel_pools'], 700)
        _heading('**Image reuse across the selected episodes**')
        _show(usage)
        _heading('**Example assignments: 5-shot development episode 0**')
        _show(episode_table(prepared, phase='dev', shots=5, episode=0))
        _heading('**Model input check**')
        _show(pd.Series(batch, name='value').to_frame())
    else:
        print('Set SHOW_DETAILS = True in Section 1 to see every table.')
    print('Data-check exports:', prepared.audit_dir)
    return prepared


def model_table(config):
    import pandas as pd
    from .models import EMBEDDING_DIMENSIONS
    return pd.DataFrame([{'model': MODEL_INFO[name][0], 'role': MODEL_INFO[name][1],
                          'in_run': name in config.models, 'embedding_dim': EMBEDDING_DIMENSIONS[name],
                          'checkpoint': MODEL_IDS[name], 'code': f'src/eurosat/models/{name}.py'}
                         for name in MODEL_IDS])


def run_or_load(config, target, storage, scratch, manifest, prepared, report_path=''):
    """Run (or resume) every selected model, or load saved results in report-only mode."""
    if report_path:
        from .evaluation import summarize
        from .reporting import create_report
        path = Path(report_path)
        saved = json.loads(path.read_text())
        results = {'output': str(path.parent), 'summary': summarize(saved['trials']),
                   'report': create_report(path, os.environ.get('EUROSAT_REPORT_OUTPUT'))}
    else:
        from .experiments import run_manifest_benchmark
        results = run_manifest_benchmark(config, storage, scratch, manifest, target=target,
                                         episode_limit=config.trials, prepared=prepared)
    print('Results folder:', results['output'])
    return results


def show_model_overview(results, show_details=False):
    """No-augmentation scores for every model and the paired gain over ResNet-18."""
    from .reporting import baseline_comparison
    figures = results['report']['figures']
    table = baseline_comparison(results)
    _heading('**Accuracy and Macro-F1 without augmentation** '
             '(fractions; gains over ResNet-18 are paired by episode, in percentage points)')
    _show(table[table.policy == 'none'].drop(columns='policy').round(4))
    _figure(figures['model_by_shot'], 950)
    if show_details:
        for name in ['classification_metrics', 'class_recall', 'confusion_matrix']:
            _figure(figures[name], 950)


def show_each_model(results):
    from .reporting import display_model_evaluation
    for name in MODEL_IDS:
        label, role = MODEL_INFO[name]
        _heading(f'#### {label} · {role}')
        display_model_evaluation(results, name)


def show_augmentation(results, config, show_details=False):
    """Paired augmentation gains, the development-selected method and per-class effects."""
    import pandas as pd
    if not [p for p in config.policies if p != 'none']:
        print("No augmentation in this run. Add methods to POLICIES in Section 1 to fill this section.")
        return
    figures = results['report']['figures']
    _heading('**Paired gain over no augmentation** (percentage points, same episodes)')
    _figure(figures['augmentation_gains'], 1000)
    columns = ['model', 'shots', 'policy', 'dev_accuracy', 'accuracy', 'paired_gain_percentage_points',
               'paired_gain_percentage_points_ci95_low', 'paired_gain_percentage_points_ci95_high',
               'fraction_nonpositive_gain']
    selected = pd.DataFrame(results['report']['selected'])
    _heading('**Method selected on development episodes, checked on test**')
    _show(selected[[c for c in columns if c in selected]].round(4))
    _figure(figures['class_augmentation_deltas'], 1000)
    if show_details:
        summary = pd.DataFrame(results['summary'])
        _heading('**All augmentation conditions** (including negative effects)')
        _show(summary[summary.policy != 'none'][[c for c in columns if c in summary]].round(4))


def output_files(results, prepared=None):
    import pandas as pd
    rows = [('Results: results.json, summary.csv', results['output']),
            ('Figures (PNG and PDF) and report tables', results['report']['output'])]
    if prepared is not None:
        rows.append(('Data-check exports', str(prepared.audit_dir)))
    return pd.DataFrame(rows, columns=['content', 'folder']).set_index('content')
