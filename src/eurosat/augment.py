"""Deterministic support-only augmentation policies."""
import math
import numbers
import random
from PIL import Image
import torch
from torchvision.transforms import functional as TF, InterpolationMode

POLICIES = {'none', 'repeat', 'flip', 'rotation', 'translation', 'scale',
            'brightness_contrast', 'noise', 'blur', 'erasing', 'cutout'}
GEOMETRIC_FILL = (124, 116, 104)


def image_tensor(path, policy, seed, strength=1.):
    if policy not in POLICIES:
        raise ValueError('Unsupported image augmentation: ' + str(policy))
    if not isinstance(seed, numbers.Integral):
        raise ValueError('Augmentation seed must be an integer')
    if (not isinstance(strength, numbers.Real) or not math.isfinite(strength)
            or not 0 < strength <= 2):
        raise ValueError('Augmentation strength must be finite and in (0, 2]')
    rng = random.Random(seed)
    with Image.open(path) as image:
        image = TF.resize(image.convert('RGB'), [224, 224], InterpolationMode.BILINEAR)
    if policy == 'flip':
        image = TF.hflip(image) if rng.random() < .5 else TF.vflip(image)
    elif policy == 'rotation':
        image = TF.rotate(image, rng.uniform(-30 * strength, 30 * strength),
                          InterpolationMode.BILINEAR, fill=GEOMETRIC_FILL)
    elif policy == 'translation':
        translation = round(22 * strength)
        image = TF.affine(image, 0,
                          [rng.randint(-translation, translation), rng.randint(-translation, translation)],
                          1., [0., 0.], InterpolationMode.BILINEAR, fill=GEOMETRIC_FILL)
    elif policy == 'scale':
        image = TF.affine(image, 0, [0, 0],
                          rng.uniform(1 - .2 * strength, 1 + .2 * strength),
                          [0., 0.], InterpolationMode.BILINEAR, fill=GEOMETRIC_FILL)
    elif policy == 'brightness_contrast':
        image = TF.adjust_contrast(TF.adjust_brightness(image, rng.uniform(1 - .3 * strength, 1 + .3 * strength)), rng.uniform(1 - .3 * strength, 1 + .3 * strength))
    elif policy == 'blur':
        image = TF.gaussian_blur(image, [5, 5], [rng.uniform(.1, 1.5 * strength)] * 2)
    tensor = TF.to_tensor(image)
    if policy == 'noise':
        generator = torch.Generator().manual_seed(seed)
        tensor = (tensor + torch.randn(tensor.shape, generator=generator) * (.05 * strength)).clamp(0, 1)
    elif policy in {'erasing', 'cutout'}:
        low = max(1, round(22 * strength ** .5))
        high = max(low, round(67 * strength ** .5))
        size = min(224, rng.randint(low, high))
        top, left = rng.randrange(225 - size), rng.randrange(225 - size)
        tensor[:, top:top + size, left:left + size] = .5 if policy == 'erasing' else 0.
    return TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
