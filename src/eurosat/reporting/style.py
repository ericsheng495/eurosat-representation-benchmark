"""Shared look for every figure: fonts, colors and small drawing helpers. Saved as PNG (notebook) and PDF (paper)."""
from pathlib import Path

import numpy as np

from ..config import AUGMENTATION_METHODS

INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#74726c', '#e4e3dd', '#c3c2b7'
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
         'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
         'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
         'text.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.facecolor': 'white',
         'savefig.facecolor': 'white'}

MODEL_SHORT = {'resnet18': 'ResNet-18', 'convnext_tiny': 'ConvNeXt-T', 'vit_b_16': 'ViT-B/16', 'dino_vits16': 'DINO-S/16'}
MODEL_COLORS = {'resnet18': '#2a78d6', 'convnext_tiny': '#eb6834', 'vit_b_16': '#1baf7a', 'dino_vits16': '#eda100'}
METHOD_LABELS = {'flip': 'flip', 'rotation': 'rotation (90°)', 'scale_up': 'scale up',
                 'brightness_contrast': 'brightness/contrast', 'noise': 'noise', 'blur': 'blur',
                 'erasing': 'erasing', 'cutout': 'cutout'}
METHOD_COLORS = dict(zip(AUGMENTATION_METHODS, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
                                                '#6250d6', '#e34948']))
REPEAT_COLOR = '#9a988f'


def colormaps():
    from matplotlib.colors import LinearSegmentedColormap
    sequential = LinearSegmentedColormap.from_list('seq', ['#eef3fb', '#9ec5f4', '#2a78d6', '#104281'])
    diverging = LinearSegmentedColormap.from_list('div', ['#b8312f', '#e34948', '#f0efec', '#2a78d6', '#1c5cab'])
    return sequential, diverging


def save_figure(figure, output, name):
    """Save as PNG and PDF; returns the PNG path."""
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'pdf']:
        figure.savefig(output / f'{name}.{extension}', dpi=300 if extension == 'pdf' else 150,
                       bbox_inches='tight', pad_inches=.06)
    plt.close(figure)
    return output / f'{name}.png'


def header(figure, title, subtitle):
    from matplotlib.transforms import offset_copy
    figure.text(0, 1, title, transform=offset_copy(figure.transFigure, fig=figure, y=20, units='points'),
                fontsize=11.5, fontweight='bold', ha='left', va='bottom')
    figure.text(0, 1, subtitle, transform=offset_copy(figure.transFigure, fig=figure, y=6, units='points'),
                fontsize=8.5, color=MUTED, ha='left', va='bottom')


def clean(ax, grid=None):
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    if grid:
        getattr(ax, f'{grid}axis').grid(True, color=GRID, lw=.8)


def panel_title(ax, text, pad=6):
    ax.set_title(text, loc='left', fontweight='bold', fontsize=10, pad=pad)


def signed(value):
    return '0.0' if abs(value) < .05 else f'{value:+.1f}'


def heat(ax, values, cmap, norm, text, white, bold=None, size=7):
    """Heatmap with a value printed in every cell (NaN cells stay blank)."""
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect='auto')
    for (r, c), value in np.ndenumerate(values):
        if value == value:
            ax.text(c, r, text(value), ha='center', va='center', fontsize=size, color='white' if white(value) else INK,
                    fontweight='bold' if bold is not None and bold[r, c] else 'normal')
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def colorbar(figure, image, axes, label, ticks=None):
    bar = figure.colorbar(image, ax=axes, shrink=.8, ticks=ticks)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8)
    bar.set_label(label, fontsize=8.5)
