"""Experiment presets and immutable dataset/checkpoint identities."""
from dataclasses import asdict, dataclass, field, fields
import copy
import math

# Bump when cached embeddings or results must not be reused. Version 2: three baselines, DINO v1,
# the new augmentation set and the wider C search.
PIPELINE_VERSION = '2'
DATA_URL = ('https://huggingface.co/datasets/torchgeo/eurosat/resolve/'
            'c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip')
DATA_MD5 = 'c8fa014336c82ac7804f0398fcb19387'
# Official DINO (v1) repository, last commit (2023-04-27). Pinned so the code never changes under us.
DINO_REVISION = '7c446df5b9f45747937fb0d72314eb9f7b66930a'
CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
# Fixed class membership from the shared split manifest:
# the six base classes are used for fine-tuning, the four novel classes for few-shot evaluation.
BASE_CLASSES = ['AnnualCrop', 'Forest', 'Highway', 'Industrial', 'Residential', 'SeaLake']
NOVEL_CLASSES = ['HerbaceousVegetation', 'Pasture', 'PermanentCrop', 'River']
# ImageNet-1K checkpoint of each model (the "pretrained" baseline, and the start of fine-tuning).
MODEL_IDS = {
    'resnet18': 'torchvision.ResNet18_Weights.IMAGENET1K_V1',
    'convnext_tiny': 'torchvision.ConvNeXt_Tiny_Weights.IMAGENET1K_V1',
    'vit_b_16': 'torchvision.ViT_B_16_Weights.IMAGENET1K_V1',
    'dino_vits16': 'facebookresearch/dino:' + DINO_REVISION + '/dino_vits16',
}
# How models are named and described in the notebook, figures and tables.
MODEL_LABELS = {'resnet18': 'ResNet-18', 'convnext_tiny': 'ConvNeXt-Tiny', 'vit_b_16': 'ViT-B/16',
                'dino_vits16': 'DINO ViT-S/16'}
MODEL_ROLES = {'resnet18': 'Standard CNN', 'convnext_tiny': 'Modern CNN', 'vit_b_16': 'Supervised Transformer',
               'dino_vits16': 'Self-supervised Transformer'}
MODEL_PRETRAINING = {'resnet18': 'ImageNet-1K, labels', 'convnext_tiny': 'ImageNet-1K, labels',
                     'vit_b_16': 'ImageNet-1K, labels', 'dino_vits16': 'ImageNet-1K, no labels'}
# Where each backbone's weights come from. Order matters: figures and tables follow it.
BASELINES = ['untrained', 'pretrained', 'finetuned']
BASELINE_LABELS = {'untrained': 'Untrained', 'pretrained': 'Pretrained', 'finetuned': 'Fine-tuned'}
UNTRAINED_SEED = 42
# Fine-tuned weights live in shared storage (not in Git), under this folder.
FINETUNED_WEIGHTS_DIR = 'cache/weights/finetuned'
# Classifier C candidates, chosen on development episodes. One per decade, 1e-3 to 1e6: on development
# episodes, accuracy peaks near 1000 and falls on both sides well before these ends.
CLASSIFIER_CS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]

# ---------------------------------------------------------------------------
# Augmentation: the single place that defines the 8 methods and how strong each one is.
# Rules: support images only; no black edges; no duplicate views; every view differs
# clearly from its original. Values act on the 224x224 image after resizing.
# The notebook's Settings cell (Section 1) shows this table and can override any value.
# 'none' (no augmentation) and 'repeat' (copies of the original) never alter pixels.
# ---------------------------------------------------------------------------
AUGMENTATION_DEFAULTS = {
    'flip': {},
    'rotation': {},
    'scale_up': {'min_zoom': 1.1, 'max_zoom': 1.3},
    'brightness_contrast': {'min_change': 0.1, 'max_change': 0.3},
    'noise': {'std': 0.05},
    'blur': {'min_sigma': 2.5, 'max_sigma': 3.5},
    'erasing': {'min_size_fraction': 0.2, 'max_size_fraction': 0.3, 'fill': 0.5},
    'cutout': {'min_size_fraction': 0.2, 'max_size_fraction': 0.3, 'fill': 0.0},
}
AUGMENTATION_METHODS = list(AUGMENTATION_DEFAULTS)
AUGMENTATION_NOTES = {
    'flip': 'mirror left-right or top-bottom; no magnitude',
    'rotation': 'quarter turn of 90, 180 or 270 degrees (lossless); no magnitude',
    'scale_up': 'zoom in on the center by a factor ~ U(min_zoom, max_zoom); no black edges',
    'brightness_contrast': 'brightness and contrast factors 1 - c or 1 + c, with c ~ U(min_change, max_change)',
    'noise': 'add Gaussian noise with this std (pixels range 0-1)',
    'blur': 'Gaussian blur with sigma ~ U(min_sigma, max_sigma) px',
    'erasing': 'gray square (fill) with side ~ U(min, max) x 224 px',
    'cutout': 'black square (fill) with side ~ U(min, max) x 224 px',
}
# Methods with a fixed set of outputs. Views of one image never repeat, so these cap the view count.
DISTINCT_VIEWS = {'flip': 2, 'rotation': 3}
# Views per support image in the view sweep (1-shot only). Flip and rotation stop at their distinct count.
SWEEP_VIEWS = [1, 2, 4, 8]
# Allowed values: (low, high). The low end is excluded except for 'fill'.
AUGMENTATION_LIMITS = {
    'min_zoom': (1., 2.), 'max_zoom': (1., 2.), 'min_change': (0., .9), 'max_change': (0., .9),
    'std': (0., 1.), 'min_sigma': (0., 10.), 'max_sigma': (0., 10.),
    'min_size_fraction': (0., 1.), 'max_size_fraction': (0., 1.), 'fill': (0., 1.),
}
RANGES = [('scale_up', 'min_zoom', 'max_zoom'), ('brightness_contrast', 'min_change', 'max_change'),
          ('blur', 'min_sigma', 'max_sigma'), ('erasing', 'min_size_fraction', 'max_size_fraction'),
          ('cutout', 'min_size_fraction', 'max_size_fraction')]


def default_augmentation():
    return copy.deepcopy(AUGMENTATION_DEFAULTS)


def validate_augmentation(settings):
    """Reject missing, misspelled or out-of-range augmentation values."""
    if set(settings) != set(AUGMENTATION_DEFAULTS):
        raise ValueError('Augmentation settings must list exactly: ' + ', '.join(AUGMENTATION_DEFAULTS))
    for policy, params in settings.items():
        if set(params) != set(AUGMENTATION_DEFAULTS[policy]):
            raise ValueError(f'{policy}: expected parameters {sorted(AUGMENTATION_DEFAULTS[policy])}')
        for name, value in params.items():
            low, high = AUGMENTATION_LIMITS[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{policy}.{name} must be a number')
            if not (low <= value if name == 'fill' else low < value) or value > high:
                bracket = '[' if name == 'fill' else '('
                raise ValueError(f'{policy}.{name}={value} is outside the allowed range {bracket}{low}, {high}]')
    for policy, low, high in RANGES:
        if settings[policy][low] > settings[policy][high]:
            raise ValueError(f'{policy}: {low} must not exceed {high}')


def augmentation_table(settings):
    """Rows for displaying each augmentation value next to its default."""
    rows = []
    for policy, params in settings.items():
        for name, value in (params or {'-': None}).items():
            default = AUGMENTATION_DEFAULTS[policy].get(name)
            rows.append({'policy': policy, 'parameter': name, 'value': value, 'default': default,
                         'changed': value != default, 'meaning': AUGMENTATION_NOTES[policy]})
    return rows


TRACKS = [1, 2, 3]  # 1 model comparison, 2 all augmentations together, 3 each augmentation alone
SMOKE_EPISODES, FULL_EPISODES = 2, 600  # Test episodes per shot. Development: 2 (smoke) or 100 (full).


@dataclass
class Config:
    """One run: which backbones, which tracks, and how many episodes."""
    mode: str = 'smoke'
    seed: int = 42
    models: list = field(default_factory=lambda: list(MODEL_IDS))
    baselines: list = field(default_factory=lambda: list(BASELINES))
    tracks: list = field(default_factory=lambda: list(TRACKS))
    view_sweep: bool = True  # Extra 1-shot run with 1, 2, 4 and 8 views per support image (figure F9).
    shots: list = field(default_factory=lambda: [1, 5, 10])
    trials: int = SMOKE_EPISODES
    augmentation: dict = field(default_factory=default_augmentation)
    classifier_cs: list = field(default_factory=lambda: list(CLASSIFIER_CS))
    batch_size: int = 64  # Images per batch when extracting embeddings.
    workers: int = 4      # Image-loading processes.
    jobs: int = -1        # Parallel classifier fits (-1 = all CPU cores).

    @classmethod
    def for_mode(cls, mode, **values):
        return cls(mode=mode, trials=FULL_EPISODES if mode == 'full' else SMOKE_EPISODES, **values)

    @classmethod
    def from_dict(cls, values):
        """Rebuild a saved configuration; settings that no longer exist are ignored."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})

    def validate(self):
        if self.mode not in {'smoke', 'full'}:
            raise ValueError('MODE must be smoke or full')
        checks = [('MODELS', self.models, list(MODEL_IDS)), ('BASELINES', self.baselines, BASELINES),
                  ('TRACKS', self.tracks, TRACKS), ('shots', self.shots, [1, 5, 10])]
        for name, chosen, allowed in checks:
            if not chosen or set(chosen) - set(allowed) or len(chosen) != len(set(chosen)):
                raise ValueError(f'{name} must be a non-empty list without repeats, from: {", ".join(map(str, allowed))}')
        if min(self.trials, self.batch_size) < 1 or self.workers < 0:
            raise ValueError('Episodes and batch size must be positive; workers must be 0 or more')
        validate_augmentation(self.augmentation)
        if not self.classifier_cs or min(self.classifier_cs) <= 0:
            raise ValueError('Classifier C values must be positive')


def config_dict(config):
    """Serialize the configuration used in experiment/cache identities."""
    return asdict(config)
