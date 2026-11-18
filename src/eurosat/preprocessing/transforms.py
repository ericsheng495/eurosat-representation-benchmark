"""Model input and support-image augmentation: 8 methods, deterministic per image and view.

Rules (research_info.md, section 4.6):
  - no black edges; no duplicate views of one image; every view differs clearly from its original
  - all models and baselines get the same views: a view depends only on the image, the method,
    the view number and the seed
Magnitudes come from AUGMENTATION_DEFAULTS in config.py or the notebook's Settings cell.
"""
import math
import random

from PIL import Image
import torch
from torchvision.transforms import functional as TF, InterpolationMode

from ..config import AUGMENTATION_DEFAULTS

IMAGE_SIZE = 224
IMAGENET_MEAN = (.485, .456, .406)
IMAGENET_STD = (.229, .224, .225)
BILINEAR = InterpolationMode.BILINEAR

# Methods with a fixed set of outputs. Each image gets its own shuffled order, and view k uses the
# k-th entry, so the views of one image never repeat.
DISCRETE = {
    'flip': [('left-right', TF.hflip), ('top-bottom', TF.vflip)],
    'rotation': [(f'{90 * k} degrees', lambda image, t=t: image.transpose(t)) for k, t in
                 [(1, Image.Transpose.ROTATE_90), (2, Image.Transpose.ROTATE_180), (3, Image.Transpose.ROTATE_270)]],
}


def distinct_option(policy, seed, view):
    """(name, function) of a flip/rotation view. Raises if the method has no more distinct views."""
    options = DISCRETE[policy]
    if view >= len(options):
        raise ValueError(f'{policy} has only {len(options)} distinct views; view {view} would repeat one')
    order = random.Random(seed).sample(range(len(options)), len(options))
    return options[order[view]]


def signed_change(rng, low, high):
    """A factor 1 - c or 1 + c with c ~ U(low, high), so it is never close to 1."""
    change = rng.uniform(low, high)
    return 1 + change if rng.random() < .5 else 1 - change


def image_tensor(path, policy='none', seed=0, params=None, view=0):
    """One normalized 3x224x224 input. 'none' gives the unaugmented image."""
    if policy != 'none' and policy not in AUGMENTATION_DEFAULTS:
        raise ValueError('Unsupported image augmentation: ' + policy)
    p = AUGMENTATION_DEFAULTS.get(policy, {}) if params is None else params
    rng = random.Random(f'{seed}:{view}')  # Same draw for the same image, method and view.
    with Image.open(path) as image:
        image = TF.resize(image.convert('RGB'), [IMAGE_SIZE, IMAGE_SIZE], BILINEAR)

    if policy in DISCRETE:
        image = distinct_option(policy, seed, view)[1](image)
    elif policy == 'scale_up':
        # Continuous zoom about the center (zoom > 1 always covers the frame, so no edges are filled).
        # Unlike an integer-pixel crop, every draw gives a different view.
        zoom = rng.uniform(p['min_zoom'], p['max_zoom'])
        image = TF.affine(image, angle=0., translate=[0, 0], scale=zoom, shear=[0., 0.], interpolation=BILINEAR)
    elif policy == 'brightness_contrast':
        image = TF.adjust_brightness(image, signed_change(rng, p['min_change'], p['max_change']))
        image = TF.adjust_contrast(image, signed_change(rng, p['min_change'], p['max_change']))
    elif policy == 'blur':
        sigma = rng.uniform(p['min_sigma'], p['max_sigma'])
        kernel = 2 * math.ceil(3 * sigma) + 1  # Covers +/-3 sigma; edges are reflected, never black.
        image = TF.gaussian_blur(image, [kernel, kernel], [sigma, sigma])

    tensor = TF.to_tensor(image)
    if policy == 'noise':
        generator = torch.Generator().manual_seed(rng.getrandbits(63))
        tensor = (tensor + torch.randn(tensor.shape, generator=generator) * p['std']).clamp(0, 1)
    elif policy in {'erasing', 'cutout'}:
        side = rng.randint(round(p['min_size_fraction'] * IMAGE_SIZE), round(p['max_size_fraction'] * IMAGE_SIZE))
        top, left = rng.randrange(IMAGE_SIZE + 1 - side), rng.randrange(IMAGE_SIZE + 1 - side)
        tensor[:, top:top + side, left:left + side] = p['fill']
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
