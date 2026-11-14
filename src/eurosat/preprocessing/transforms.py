"""Deterministic support-only augmentation policies.

Magnitudes come from AUGMENTATION_DEFAULTS in config.py (or the notebook's
AUGMENTATION overrides). With default values, every policy except rotation is
pixel-identical to the earlier fixed-strength implementation, so those cached
embeddings stay valid. Rotation uses 90-degree steps only (AUGMENTATION_VERSIONS).
"""
import random
from PIL import Image
import torch
from torchvision.transforms import functional as TF, InterpolationMode
from ..config import AUGMENTATION_DEFAULTS

IMAGE_SIZE = 224
IMAGENET_MEAN = (.485, .456, .406)
IMAGENET_STD = (.229, .224, .225)
# Overhead imagery has no "up": quarter turns keep every pixel and add no empty corners.
QUARTER_TURNS = [Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270]


def image_tensor(path, policy, seed, params=None):
    """Load one image; for an augmentation policy, draw one random view with `params`."""
    if policy != 'none' and policy not in AUGMENTATION_DEFAULTS:
        raise ValueError('Unsupported image augmentation: ' + policy)
    p = AUGMENTATION_DEFAULTS.get(policy, {}) if params is None else params
    rng = random.Random(seed)
    with Image.open(path) as image:
        image = TF.resize(image.convert('RGB'), [IMAGE_SIZE, IMAGE_SIZE], InterpolationMode.BILINEAR)
    if policy == 'flip':
        image = TF.hflip(image) if rng.random() < .5 else TF.vflip(image)
    elif policy == 'rotation':
        image = image.transpose(rng.choice(QUARTER_TURNS))
    elif policy == 'translation':
        shift = round(p['max_shift_fraction'] * IMAGE_SIZE)
        image = TF.affine(image, 0, [rng.randint(-shift, shift), rng.randint(-shift, shift)], 1., [0., 0.],
                          InterpolationMode.BILINEAR)
    elif policy == 'scale':
        zoom = p['max_zoom']
        image = TF.affine(image, 0, [0, 0], rng.uniform(1 - zoom, 1 + zoom), [0., 0.], InterpolationMode.BILINEAR)
    elif policy == 'brightness_contrast':
        b, c = p['max_brightness'], p['max_contrast']
        image = TF.adjust_contrast(TF.adjust_brightness(image, rng.uniform(1 - b, 1 + b)), rng.uniform(1 - c, 1 + c))
    elif policy == 'blur':
        kernel = int(p['kernel_size'])
        image = TF.gaussian_blur(image, [kernel, kernel], [rng.uniform(p['min_sigma'], p['max_sigma'])] * 2)
    tensor = TF.to_tensor(image)
    if policy == 'noise':
        generator = torch.Generator().manual_seed(seed)
        tensor = (tensor + torch.randn(tensor.shape, generator=generator) * p['std']).clamp(0, 1)
    elif policy in {'erasing', 'cutout'}:
        size = rng.randint(round(p['min_size_fraction'] * IMAGE_SIZE), round(p['max_size_fraction'] * IMAGE_SIZE))
        top, left = rng.randrange(IMAGE_SIZE + 1 - size), rng.randrange(IMAGE_SIZE + 1 - size)
        tensor[:, top:top + size, left:left + size] = p['fill']
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
