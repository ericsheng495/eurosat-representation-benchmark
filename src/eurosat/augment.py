"""Deterministic support-only augmentation policies."""
import random
from PIL import Image
import torch
from torchvision.transforms import functional as TF, InterpolationMode


def image_tensor(path, policy, seed):
    rng = random.Random(seed)
    with Image.open(path) as image:
        image = TF.resize(image.convert('RGB'), [224, 224], InterpolationMode.BILINEAR)
    if policy == 'flip':
        image = TF.hflip(image) if rng.random() < .5 else TF.vflip(image)
    elif policy == 'rotation':
        image = TF.rotate(image, rng.uniform(-30, 30), InterpolationMode.BILINEAR)
    elif policy == 'translation':
        image = TF.affine(image, 0, [rng.randint(-22, 22), rng.randint(-22, 22)], 1., [0., 0.],
                          InterpolationMode.BILINEAR)
    elif policy == 'scale':
        image = TF.affine(image, 0, [0, 0], rng.uniform(.8, 1.2), [0., 0.], InterpolationMode.BILINEAR)
    elif policy == 'brightness_contrast':
        image = TF.adjust_contrast(TF.adjust_brightness(image, rng.uniform(.7, 1.3)), rng.uniform(.7, 1.3))
    elif policy == 'blur':
        image = TF.gaussian_blur(image, [5, 5], [rng.uniform(.1, 1.5)] * 2)
    elif policy not in {'none', 'noise', 'erasing', 'cutout'}:
        raise ValueError('Unsupported image augmentation: ' + policy)
    tensor = TF.to_tensor(image)
    if policy == 'noise':
        generator = torch.Generator().manual_seed(seed)
        tensor = (tensor + torch.randn(tensor.shape, generator=generator) * .05).clamp(0, 1)
    elif policy in {'erasing', 'cutout'}:
        size = rng.randint(22, 67)
        top, left = rng.randrange(225 - size), rng.randrange(225 - size)
        tensor[:, top:top + size, left:left + size] = .5 if policy == 'erasing' else 0.
    return TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
