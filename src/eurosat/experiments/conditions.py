"""What the few-shot classifier is trained on, per original support image.

  track 1   none            the original
  track 2   all             the original + one view from each of the 8 methods     control: repeat-9
  track 3   <method>        the original + one view from that method               control: repeat-2
  sweep     <method>-<v>    the original + v views from that method (1-shot only)  control: repeat-<v+1>

A "repeat-<n>" control copies each original n times: as many rows as its augmented partner, but no new pixels.
"none" always runs: every gain is measured against it on the same episodes.
"""
from dataclasses import dataclass

from ..config import AUGMENTATION_METHODS, DISTINCT_VIEWS, SWEEP_VIEWS


@dataclass(frozen=True)
class Condition:
    name: str
    track: str        # '1', '2', '3' or 'sweep'
    copies: int       # rows per original support image taken from the original itself
    views: tuple = ()  # ((method, view number), ...) rows added per original support image
    control: bool = False

    @property
    def rows(self):
        return self.copies + len(self.views)

    @property
    def methods(self):
        return sorted({method for method, _ in self.views})


def sweep_views(method):
    """View counts in the sweep. Flip stops at 2 and rotation at 3: they have no more distinct views."""
    limit = DISTINCT_VIEWS.get(method)
    if limit is None:
        return list(SWEEP_VIEWS)
    return sorted({v for v in SWEEP_VIEWS if v <= limit} | {limit})


def build_conditions(tracks, view_sweep, shots):
    """[(Condition, shot budgets)] for one run, in a fixed order."""
    plan = {}

    def add(condition, condition_shots):
        if condition.name in plan:
            plan[condition.name] = (condition, sorted(set(plan[condition.name][1]) | set(condition_shots)))
        else:
            plan[condition.name] = (condition, sorted(condition_shots))

    add(Condition('none', '1', 1), shots)
    if 2 in tracks:
        add(Condition('all', '2', 1, tuple((m, 0) for m in AUGMENTATION_METHODS)), shots)
        add(Condition('repeat-9', '2', 9, control=True), shots)
    if 3 in tracks:
        for method in AUGMENTATION_METHODS:
            add(Condition(method, '3', 1, ((method, 0),)), shots)
        add(Condition('repeat-2', '3', 2, control=True), shots)
    if view_sweep and 1 in shots:
        for method in AUGMENTATION_METHODS:
            for views in sweep_views(method):
                name = method if views == 1 else f'{method}-{views}'
                add(Condition(name, '3' if views == 1 else 'sweep', 1, tuple((method, v) for v in range(views))), [1])
                rows = views + 1
                add(Condition(f'repeat-{rows}', {2: '3', 9: '2'}.get(rows, 'sweep'), rows, control=True), [1])
    return list(plan.values())


def needed_views(plan):
    """{(method, view): shot budgets} whose augmented embeddings the plan needs."""
    needed = {}
    for condition, condition_shots in plan:
        for item in condition.views:
            needed.setdefault(item, set()).update(condition_shots)
    return {item: sorted(shots) for item, shots in sorted(needed.items())}
