"""The notebook's helper: one object per run, one method per notebook section.

    run = nb.Run.from_settings(globals())   # 2 Setup: read the Section 1 settings, check the device
    run.check_data()                        # 3 Data check
    run.fine_tune()                         # 4 Fine-tune
    run.run()                               # 5 Run, then make the report figures and tables
    run.show_models()                       # 6 Results: models and baselines
    run.show_augmentation()                 # 7 Results: augmentation
    run.show_checks()                       # 8 Checks and files

With REPORT_RESULTS set, sections 3-5 load saved results instead of computing anything.
"""
import copy
import json
import os
from pathlib import Path

from .config import (AUGMENTATION_DEFAULTS, BASELINES, MODEL_IDS, MODEL_LABELS, Config, augmentation_table)
from .runtime import check_device

# Automated runs (scripts/execute_notebook.py) override Section 1 settings with these variables.
OVERRIDES = {
    'MODE': ('EUROSAT_MODE', str), 'MODELS': ('EUROSAT_MODELS', str),
    'BASELINES': ('EUROSAT_BASELINES', lambda raw: raw.split(',')),
    'TRACKS': ('EUROSAT_TRACKS', lambda raw: [int(t) for t in raw.split(',')]),
    'VIEW_SWEEP': ('EUROSAT_VIEW_SWEEP', lambda raw: raw.lower() in {'1', 'true', 'yes'}),
    'MANIFEST_FILE': ('EUROSAT_SPLITS', str), 'REPORT_RESULTS': ('EUROSAT_REPORT_RESULTS', str),
}


def _display(*items):
    from IPython.display import display
    display(*items)


def _markdown(text):
    from IPython.display import Markdown
    _display(Markdown(text))


def _image(path, width=1000):
    from IPython.display import Image
    _display(Image(filename=str(path), width=width))


def _names(raw, allowed):
    """'all', 'a,b' or a list -> a list of names."""
    if raw in ('all', ['all']):
        return list(allowed)
    return [name.strip() for name in (raw.split(',') if isinstance(raw, str) else raw)]


class Run:
    def __init__(self, config, target, storage, scratch, manifest, report_path='', show_details=False):
        self.config, self.target, self.storage, self.scratch = config, target, Path(storage), Path(scratch)
        self.manifest, self.report_path, self.show_details = Path(manifest), report_path, show_details
        self.prepared = self.results = self.report = None

    @property
    def report_only(self):
        return bool(self.report_path)

    @classmethod
    def from_settings(cls, settings):
        """Section 2: build the run from the notebook's Section 1 variables and show what it will do."""
        s = dict(settings)
        for name, (variable, convert) in OVERRIDES.items():
            if os.environ.get(variable):
                s[name] = convert(os.environ[variable])
                print(f'{name} = {s[name]!r} (from {variable})')
        if s['REPORT_RESULTS']:
            config = Config.from_dict(json.loads(Path(s['REPORT_RESULTS']).read_text())['config'])
        else:
            config = Config.for_mode(s['MODE'], models=_names(s['MODELS'], MODEL_IDS),
                                     baselines=_names(s['BASELINES'], BASELINES), tracks=list(s['TRACKS']),
                                     view_sweep=bool(s['VIEW_SWEEP']), augmentation=copy.deepcopy(s['AUGMENTATION']))
            config.trials = int(os.environ.get('EUROSAT_TRIALS', config.trials))
        config.validate()
        storage = Path(s['STORAGE'])
        manifest = Path(s['MANIFEST_FILE']) if s['MANIFEST_FILE'] else storage / 'splits_manifest.json'
        run = cls(config, s['TARGET'], storage, s['SCRATCH'], manifest, s['REPORT_RESULTS'], s['SHOW_DETAILS'])
        if not run.report_only:
            check_device(run.target)
        _display(run.summary())
        if run.show_details:
            _display(run.augmentation_table())
        return run

    def summary(self):
        import pandas as pd
        c = self.config
        changed = [f'{method}.{name} = {value}' for method, params in c.augmentation.items()
                   for name, value in params.items() if value != AUGMENTATION_DEFAULTS[method][name]]
        rows = [('Mode', 'smoke (quick code check)' if c.mode == 'smoke' else 'full (the experiment)'),
                ('Models', ', '.join(MODEL_LABELS[m] for m in c.models)),
                ('Baselines', ', '.join(c.baselines)),
                ('Tracks', ', '.join(map(str, c.tracks)) + (' + view sweep' if c.view_sweep else '')),
                ('Episodes per shot', f"{2 if c.mode == 'smoke' else 100} development + {c.trials} test"),
                ('Augmentation', 'defaults' if not changed else 'changed: ' + '; '.join(changed)),
                ('Storage', str(self.storage))]
        if self.report_only:
            rows.insert(0, ('Report only', self.report_path))
        return pd.DataFrame(rows, columns=['setting', 'value']).set_index('setting')

    def augmentation_table(self):
        import pandas as pd
        table = pd.DataFrame(augmentation_table(self.config.augmentation))
        table[['value', 'default']] = table[['value', 'default']].astype(object)
        table.loc[table.parameter == '-', ['value', 'default']] = '—'  # flip and rotation have no strength.
        return table

    def check_data(self):
        """Section 3: verify the archive, the shared split and this run's episodes."""
        if self.report_only:
            print('Report only: data check skipped.')
            return
        import pandas as pd
        from .preprocessing import load_source, select_episodes, validate_manifest
        from .preprocessing.audit import data_figures, episode_table, inspect_batch, usage_table

        source = load_source(self.storage, self.scratch)
        validated = validate_manifest(self.manifest, source)
        self.prepared = prepared = select_episodes(source, validated, self.config, self.storage)
        batch, usage, figures = inspect_batch(prepared, self.config), usage_table(prepared), data_figures(prepared, validated)
        integrity, dev = source.integrity, len(next(iter(prepared.episodes['dev'].values())))
        rows = [('Archive checksum', 'EuroSAT.zip matches the published MD5'),
                ('Extracted files', f"{integrity['files_crc_verified']:,} files match the archive"),
                ('Images', f"{integrity['corrupt_images']} corrupt, {integrity['decoded_pixel_duplicates']} duplicates"),
                *[(check['check'], 'every pool and episode') for check in validated.checks],
                ('Episodes selected', f'{dev} development + {self.config.trials} test per shot, '
                                      f'{len(prepared.records):,} images'),
                ('Model input', f"{tuple(batch['batch_shape'][1:])}, finite, identical on re-read")]
        _display(pd.DataFrame(rows, columns=['check (all passed)', 'detail']).set_index('check (all passed)'))
        _image(figures['support_examples'], 650)
        if self.show_details:
            _display(pd.DataFrame(source.inventory), pd.DataFrame(validated.pools),
                     pd.DataFrame(validated.episode_counts), usage, episode_table(prepared, 'dev', 5, 0),
                     pd.Series(batch, name='value').to_frame())
            _image(figures['novel_pools'], 650)

    def fine_tune(self):
        """Section 4: train the fine-tuned baseline. Finished models and learning rates are reused."""
        if self.report_only or 'finetuned' not in self.config.baselines:
            print('Skipped: report only, or no fine-tuned baseline in BASELINES.')
            return
        import pandas as pd
        from .training import fine_tune
        smoke = self.config.mode == 'smoke'
        if smoke:
            print('Smoke mode: 2 batches per epoch into a separate folder, only to check the code.')
        records = fine_tune(self.config.models, self.storage, self.scratch, self.manifest,
                            workers=self.config.workers, smoke=smoke)
        _display(pd.DataFrame([{'model': MODEL_LABELS[r['model']], 'learning rate': r['chosen_learning_rate'],
                                'best epoch': r['best_epoch'], 'val accuracy': round(r['val_accuracy'], 4),
                                'test accuracy (6 classes)': round(r['test_accuracy'], 4),
                                'minutes': round(sum(run['minutes'] for run in r['runs']), 1)}
                               for r in records]).set_index('model'))

    def run(self):
        """Section 5: run or resume every baseline x model x condition, then make the report."""
        from .experiments import load, run
        from .reporting import build_report
        path = self.report_path or run(self.config, self.storage, self.scratch, self.manifest, self.prepared,
                                       self.target)['results_path']
        self.results = load(path)
        self.report = build_report(self.results, self.prepared, self.config)

    def _show(self, main, supplementary):
        for item in main + (supplementary if self.show_details else []):
            title = f'**{item} · {self.report.title(item)}**'
            if item in self.report.figures:
                _markdown(title)
                _image(self.report.figures[item])
            elif item in self.report.tables:
                _markdown(title)
                _display(self.report.tables[item])
            else:
                _markdown(f'{title}: not made; it {self.report.skipped[item]}.')
        if supplementary and not self.show_details:
            print(f'Also saved: {", ".join(supplementary)} (shown with SHOW_DETAILS = True).')

    def show_models(self):
        """Section 6: T1, T5, F2, F3, F11 (+ F4, F5, T2)."""
        self._show(['T1', 'T5', 'F2', 'F3', 'F11'], ['F4', 'F5', 'T2'])

    def show_augmentation(self):
        """Section 7: F6, F8, T3 (+ F1, F7, F9, F10)."""
        self._show(['F6', 'F8', 'T3'], ['F1', 'F7', 'F9', 'F10'])

    def show_checks(self):
        """Section 8: classifier checks, labels used (T4) and where everything is saved."""
        import pandas as pd
        groups = self.results.groups
        cs = ', '.join(f'{c:g}' for c in self.config.classifier_cs)
        print(f'C chosen at the edge of {{{cs}}}: {int(groups.c_at_edge.sum())} of {len(groups)} groups')
        print(f'Classifier fits that hit the iteration limit: {int(groups.nonconverged_fits.sum())}')
        self._show(['T4'], [])
        rows = [('Figures (PNG, PDF) and tables (CSV)', self.report.output),
                ('Results list', self.results.path), ('Per-episode results', self.storage / 'cache/results')]
        if self.prepared is not None:
            rows.append(('Data-check exports', self.prepared.audit_dir))
        _display(pd.DataFrame([(a, str(b)) for a, b in rows], columns=['what', 'where']).set_index('what'))
