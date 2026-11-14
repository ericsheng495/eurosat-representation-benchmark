"""Experiment presets and immutable dataset/checkpoint identities."""
from dataclasses import asdict, dataclass, field
import copy
import math

PIPELINE_VERSION = '1'
DATA_URL = ('https://huggingface.co/datasets/torchgeo/eurosat/resolve/'
            'c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip')
DATA_MD5 = 'c8fa014336c82ac7804f0398fcb19387'
DINO_REVISION = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
# Fixed class membership for four-way evaluation from the shared split manifest.
BASE_CLASSES = ['AnnualCrop', 'Forest', 'Highway', 'Industrial', 'Residential', 'SeaLake']
NOVEL_CLASSES = ['HerbaceousVegetation', 'Pasture', 'PermanentCrop', 'River']
MODEL_IDS = {
    'resnet18': 'torchvision.ResNet18_Weights.IMAGENET1K_V1',
    'convnext_tiny': 'torchvision.ConvNeXt_Tiny_Weights.IMAGENET1K_V1',
    'vit_b_16': 'torchvision.ViT_B_16_Weights.IMAGENET1K_V1',
    'dinov2_vits14': 'facebookresearch/dinov2:' + DINO_REVISION + '/dinov2_vits14',
}
POLICIES = ['none', 'repeat', 'flip', 'rotation', 'translation', 'scale',
            'brightness_contrast', 'noise', 'blur', 'erasing']
OPTIONAL_POLICIES = ['cutout']

# ---------------------------------------------------------------------------
# Augmentation magnitudes: the single place that defines how strong each
# augmentation is. Values act on the 224x224 image after resizing, and each
# augmented view draws a new random value inside these limits.
# The notebook's Settings cell (Section 1) shows this table and can override any value.
# 'none' and 'repeat' have no magnitudes: they never alter pixels.
# ---------------------------------------------------------------------------
AUGMENTATION_DEFAULTS = {
    'flip': {},
    'rotation': {},
    'translation': {'max_shift_fraction': 0.1},
    'scale': {'max_zoom': 0.2},
    'brightness_contrast': {'max_brightness': 0.3, 'max_contrast': 0.3},
    'noise': {'std': 0.05},
    'blur': {'kernel_size': 5, 'min_sigma': 0.1, 'max_sigma': 1.5},
    'erasing': {'min_size_fraction': 0.1, 'max_size_fraction': 0.3, 'fill': 0.5},
    'cutout': {'min_size_fraction': 0.1, 'max_size_fraction': 0.3, 'fill': 0.0},
}
AUGMENTATION_NOTES = {
    'flip': 'mirror left-right or top-bottom (50/50); no magnitude',
    'rotation': 'quarter turn of 90, 180 or 270 degrees (lossless); no magnitude',
    'translation': 'x and y shift ~ U(-f, +f) x 224 px each; empty border is black',
    'scale': 'zoom factor ~ U(1 - max_zoom, 1 + max_zoom)',
    'brightness_contrast': 'brightness and contrast factors ~ U(1 - max, 1 + max)',
    'noise': 'add Gaussian noise with this std (pixels range 0-1)',
    'blur': 'Gaussian blur; sigma ~ U(min_sigma, max_sigma) px; odd kernel_size',
    'erasing': 'square with side ~ U(min, max) x 224 px, filled with gray (fill)',
    'cutout': 'same square as erasing, filled with black (fill)',
}
# Allowed values: (low, high). The low end is excluded except for 'fill'.
AUGMENTATION_LIMITS = {
    'max_shift_fraction': (0., .5), 'max_zoom': (0., .9),
    'max_brightness': (0., 1.), 'max_contrast': (0., 1.), 'std': (0., 1.),
    'kernel_size': (1., 31.), 'min_sigma': (0., 10.), 'max_sigma': (0., 10.),
    'min_size_fraction': (0., 1.), 'max_size_fraction': (0., 1.), 'fill': (0., 1.),
}

# Bump a policy's version whenever its pixel output changes, so cached embeddings
# from the earlier behavior are never reused. Rotation v2 = 90-degree steps only
# (v1 tilted by up to +/-30 degrees and left black corners).
AUGMENTATION_VERSIONS = {'rotation': 2}


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
    blur = settings['blur']
    if blur['kernel_size'] != int(blur['kernel_size']) or blur['kernel_size'] % 2 == 0:
        raise ValueError('blur.kernel_size must be an odd integer')
    for policy, low, high in [('blur', 'min_sigma', 'max_sigma'),
                              ('erasing', 'min_size_fraction', 'max_size_fraction'),
                              ('cutout', 'min_size_fraction', 'max_size_fraction')]:
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


@dataclass
class Config:
    mode: str = 'smoke'
    data_protocol: str = 'base_novel'
    seed: int = 42
    models: list = field(default_factory=lambda: list(MODEL_IDS))
    policies: list = field(default_factory=lambda: ['none'])
    shots: list = field(default_factory=lambda: [1, 5, 10])
    trials: int = 2
    views: int = 1
    per_class_limit: int | None = None
    batch_size: int = 32
    chunk_size: int = 256
    augmentation: dict = field(default_factory=default_augmentation)
    classifier_cs: list = field(default_factory=lambda: [0.1, 1.0, 10.0])

    @classmethod
    def full(cls):
        return cls(mode='full', models=list(MODEL_IDS), policies=['none'],
                   trials=600, views=4, per_class_limit=None)

    @classmethod
    def from_dict(cls, values):
        """Rebuild a saved configuration, including results saved before augmentation settings."""
        values = dict(values)
        if values.pop('policy_strengths', None):
            raise ValueError('These results used legacy strength multipliers; rerun with augmentation settings')
        values.setdefault('augmentation', default_augmentation())  # Older runs used these defaults.
        return cls(**values)

    def validate(self):
        if self.data_protocol != 'base_novel':
            raise ValueError('Use the fixed base_novel manifest protocol')
        if self.mode not in {'smoke', 'full'}:
            raise ValueError('mode must be smoke or full')
        if not self.models or set(self.models) - MODEL_IDS.keys():
            raise ValueError('Choose supported models: ' + ', '.join(MODEL_IDS))
        if 'none' not in self.policies or set(self.policies) - set(POLICIES + OPTIONAL_POLICIES):
            raise ValueError('Policies must include none and contain supported names')
        if not self.shots or min(self.shots) < 1 or min(self.trials, self.views, self.batch_size, self.chunk_size) < 1:
            raise ValueError('Shot budgets, trials, views and batch/chunk sizes must be positive')
        if any(len(values) != len(set(values)) for values in [self.models, self.policies, self.shots]):
            raise ValueError('Models, policies and shot budgets must be unique')
        validate_augmentation(self.augmentation)
        if not self.classifier_cs or min(self.classifier_cs) <= 0:
            raise ValueError('Classifier C values must be positive')


def config_dict(config):
    """Serialize the configuration used in experiment/cache identities."""
    return asdict(config)
