"""Combine compatible teammate exports without mixing experimental protocols."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path

from .cache import atomic, key, write_json
from .config import MODEL_IDS
from .evaluation import summarize


def comparison_identity(manifest):
    comparable = deepcopy(manifest)
    comparable['config'].pop('models')
    comparable.pop('checkpoints')
    return key(comparable)


def merge_results(inputs, output):
    """Merge completed model assignments with identical data, trials and settings."""
    if not inputs:
        raise ValueError('Supply at least one results.json export')
    reports = [json.loads(Path(path).read_text()) for path in inputs]
    identity = comparison_identity(reports[0]['manifest'])
    rows, seen, checkpoints = [], set(), {}
    for report in reports:
        manifest = report['manifest']
        if comparison_identity(manifest) != identity:
            raise ValueError('Cannot combine different datasets, trials, software versions or non-model settings')
        config = manifest['config']
        if set(manifest['checkpoints']) != set(config['models']):
            raise ValueError('Checkpoint identities must cover exactly the exported models')
        expected = {(model, policy, shot, trial) for model in config['models']
                    for policy in config['policies'] for shot in config['shots']
                    for trial in range(config['trials'])}
        received = [(r['model'], r['policy'], r['shots'], r['trial']) for r in report['trials']]
        if len(received) != len(expected) or set(received) != expected:
            raise ValueError('A teammate export contains incomplete or duplicated trials')
        if seen & set(received):
            raise ValueError('The same model/policy/shot/trial appears in multiple exports')
        for model, checkpoint in manifest['checkpoints'].items():
            if MODEL_IDS.get(model) != checkpoint:
                raise ValueError('Unrecognized checkpoint identity for ' + model)
        checkpoints.update(manifest['checkpoints'])
        seen.update(received)
        rows.extend(report['trials'])
    manifest = deepcopy(reports[0]['manifest'])
    manifest['config']['models'] = [m for m in MODEL_IDS if m in checkpoints]
    manifest['checkpoints'] = checkpoints
    summary = summarize(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'results.json', {'manifest': manifest, 'trials': rows, 'summary': summary})
    stream = io.StringIO()
    writer = csv.DictWriter(stream, lineterminator='\n', fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
    atomic(output / 'summary.csv', lambda p: p.write_text(stream.getvalue()))
    print(f'Merged {len(rows)} trials across {len(checkpoints)} models into {output}', flush=True)
    return {'output': str(output), 'summary': summary, 'models': manifest['config']['models']}
