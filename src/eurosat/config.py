"""Experiment presets and immutable dataset/checkpoint identities."""
from dataclasses import dataclass, field
import math

PIPELINE_VERSION = '1'
DATA_URL = ('https://huggingface.co/datasets/torchgeo/eurosat/resolve/'
            'c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip')
DATA_MD5 = 'c8fa014336c82ac7804f0398fcb19387'
DINO_REVISION = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
MODEL_IDS = {
    'resnet18': 'torchvision.ResNet18_Weights.IMAGENET1K_V1',
    'convnext_tiny': 'torchvision.ConvNeXt_Tiny_Weights.IMAGENET1K_V1',
    'vit_b_16': 'torchvision.ViT_B_16_Weights.IMAGENET1K_V1',
    'dinov2_vits14': 'facebookresearch/dinov2:' + DINO_REVISION + '/dinov2_vits14',
}
POLICIES = ['none', 'repeat', 'flip', 'rotation', 'translation', 'scale',
            'brightness_contrast', 'noise', 'blur', 'erasing']
OPTIONAL_POLICIES = ['cutout']


@dataclass
class Config:
    mode: str = 'smoke'
    seed: int = 42
    models: list = field(default_factory=lambda: ['resnet18'])
    policies: list = field(default_factory=lambda: ['none', 'repeat', 'flip'])
    shots: list = field(default_factory=lambda: [1, 5, 10])
    trials: int = 1
    views: int = 1
    per_class_limit: int | None = 40
    batch_size: int = 32
    chunk_size: int = 256
    policy_strengths: dict = field(default_factory=dict)
    classifier_cs: list = field(default_factory=lambda: [0.1, 1.0, 10.0])

    @classmethod
    def full(cls):
        return cls(mode='full', models=list(MODEL_IDS), policies=POLICIES.copy(),
                   trials=20, views=4, per_class_limit=None)

    def validate(self):
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
        if set(self.policy_strengths) - set(POLICIES[2:] + OPTIONAL_POLICIES):
            raise ValueError('Strengths must name supported augmentation policies')
        if any(not math.isfinite(s) or not 0 < s <= 2 for s in self.policy_strengths.values()):
            raise ValueError('Augmentation strengths must be finite and in (0, 2]')
        if not self.classifier_cs or min(self.classifier_cs) <= 0:
            raise ValueError('Classifier C values must be positive')
