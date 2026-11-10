"""Deterministic support-only augmentation policies."""
import random
from PIL import Image
import torch
from torchvision.transforms import functional as TF, InterpolationMode

IMAGE_SIZE = 224
IMAGENET_MEAN = (.485, .456, .406)
IMAGENET_STD = (.229, .224, .225)

def image_tensor(path, policy, seed, strength=1.):
    rng = random.Random(seed)
    with Image.open(path) as image:
        image = TF.resize(image.convert('RGB'), [IMAGE_SIZE, IMAGE_SIZE], InterpolationMode.BILINEAR)
    if policy == 'flip':
        image = TF.hflip(image) if rng.random() < .5 else TF.vflip(image)
    elif policy == 'rotation':
        image = TF.rotate(image, rng.uniform(-30 * strength, 30 * strength), InterpolationMode.BILINEAR)
    elif policy == 'translation':
        image = TF.affine(image, 0, [rng.randint(-round(22 * strength), round(22 * strength)), rng.randint(-round(22 * strength), round(22 * strength))], 1., [0., 0.],
                          InterpolationMode.BILINEAR)
    elif policy == 'scale':
        image = TF.affine(image, 0, [0, 0], rng.uniform(1 - .2 * strength, 1 + .2 * strength), [0., 0.], InterpolationMode.BILINEAR)
    elif policy == 'brightness_contrast':
        image = TF.adjust_contrast(TF.adjust_brightness(image, rng.uniform(1 - .3 * strength, 1 + .3 * strength)), rng.uniform(1 - .3 * strength, 1 + .3 * strength))
    elif policy == 'blur':
        image = TF.gaussian_blur(image, [5, 5], [rng.uniform(.1, 1.5 * strength)] * 2)
    elif policy not in {'none', 'noise', 'erasing', 'cutout'}:
        raise ValueError('Unsupported image augmentation: ' + policy)
    tensor = TF.to_tensor(image)
    if policy == 'noise':
        generator = torch.Generator().manual_seed(seed)
        tensor = (tensor + torch.randn(tensor.shape, generator=generator) * (.05 * strength)).clamp(0, 1)
    elif policy in {'erasing', 'cutout'}:
        size = rng.randint(round(22 * strength ** .5), round(67 * strength ** .5))
        top, left = rng.randrange(225 - size), rng.randrange(225 - size)
        tensor[:, top:top + size, left:left + size] = .5 if policy == 'erasing' else 0.
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
