from typing import Iterable, Union, Tuple

from matplotlib import cm, pyplot as plt
# Axes3d's required for 3d plots, even though it's not specifically required.
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def _set_grid_axes(ax):
    ax.grid(True)

    ax.axhline(y=0, linewidth=1.5, color='darkslategrey')
    ax.axvline(x=0, linewidth=1.5, color='darkslategrey')


def _set_misc(fig, ax, title: str, grid: bool, equal_aspect: bool) -> None:
    if grid:
        _set_grid_axes(ax)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
        top_margin = .95
    else:
        top_margin = 1

    if equal_aspect:
        ax.set_aspect('equal')

    plt.tight_layout(rect=(0, 0, 1, top_margin))
    # fig.patch.set_facecolor('ghostwhite')


def _show_or_return(ax, show):
    if show:
        plt.show()
    else:
        return ax


def plot2(args, marker='b-', linewidth: float=2.0, grid: bool=False, color: str=None,
         title: str=None, equal_aspect: bool=False, style: str=None, show: bool=True) -> None:
    """Create a 2d plot; wrapper for plt.plot. Can return an ax object, or show
    a plot directly."""

    if style:
        plt.style.use(style)  # style must be set before setting fix, ax.

    fig, ax = plt.subplots()

    ax.plot(args, marker, color=color, linewidth=linewidth)

    if equal_aspect:
        ax.set_aspect('equal')

    _set_misc(fig, ax, title, grid, equal_aspect)

    return _show_or_return(ax, show)


def imshow(image: np.ndarray, extent: Tuple=None, colorbar=True, show=True):
    fig, ax = plt.subplots()
    cax = ax.imshow(image, extent=extent)
    if colorbar:
        fig.colorbar(cax)

    return _show_or_return(ax, show)
