"""Compute selection and software provenance."""
from pathlib import Path
import os
import numpy as np
import sklearn
import torch
import torchvision


def environment():
    return {'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
            'numpy': np.__version__, 'sklearn': sklearn.__version__}


def check_device(target='colab_gpu'):
    if target not in {'colab_gpu', 'local'}:
        raise ValueError('target must be colab_gpu or local')
    if target == 'colab_gpu':
        if not Path('/content').is_dir() or not torch.cuda.is_available():
            raise RuntimeError('Select a Colab GPU server (T4 or L4), not a local/CPU/TPU '
                               'kernel. In VS Code: Select Another Kernel > Colab > '
                               'New Colab Server > GPU > Python 3 (ipykernel).')
        device = torch.device('cuda')
        label = torch.cuda.get_device_name(0)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else
                              'mps' if torch.backends.mps.is_available() else 'cpu')
        label = str(device)
    if device.type == 'cpu':
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    if (torch.ones(1, device=device) + 1).cpu().item() != 2:
        raise RuntimeError('Device operation check failed')
    print(f'Compute: {label} [{device}] — device check passed', flush=True)
    return device
