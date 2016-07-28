from functools import wraps
from typing import Callable, Tuple

from matplotlib import cm, pyplot as plt
# Axes3d's required for 3d plots, even though it's not specifically required.
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


DEFAULT_STYLE = 'seaborn-deep'


# Todo broken atm.
def show2(f):
    @wraps(f)
    def inner(*args, **kwargs):
        if kwargs['show']:
            plt.show()
        else:
            return f(*args, **kwargs)
    return inner


def from_sympy():
    pass


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

    fig.patch.set_facecolor('ghostwhite')


def _show_or_return(ax, show):
    if show:
        plt.show()
    else:
        return ax


def plot(f: Callable[[float], float], x_min: float, x_max: float,
         title: str=None, grid=True, show=True, equal_aspect=False,
         color: str=None, resolution=1e5, style: str=DEFAULT_STYLE) -> None:
    """One input, one output."""

    x = np.linspace(x_min, x_max, resolution)
    y = f(x)

    # Style seems to require a reset, or some properties from previous styles stick.
    plt.style.use('classic')
    plt.style.use(style)  # style must be set before setting fix, ax.
    fig, ax = plt.subplots()

    ax.plot(x, y, color=color)
    if equal_aspect:
        ax.set_aspect('equal')

    _set_misc(fig, ax, title, grid, equal_aspect)
    _show_or_return(ax, show)


def parametric(f: Callable[[float], Tuple[float, float]], t_min: float,
               t_max: float, title: str=None, grid=True, show=True,
               color=None, equal_aspect=False, resolution=1e5, style: str=DEFAULT_STYLE)-> None:
    """One input, two outputs (2d plot), three outputs (3d plot)."""

    t = np.linspace(t_min, t_max, resolution)
    outputs = f(t)

    # Style seems to require a reset, or some properties from previous styles stick.
    plt.style.use('classic')
    plt.style.use(style)  # style must be set before setting fix, ax.
    if len(outputs) == 2:
        fig, ax = _parametric2d(*outputs, color)
    elif len(outputs) == 3:
        fig, ax = _parametric3d(*outputs, color)
        grid = False  # The grid and axes doesn't work the same way on mpl's 3d plots.
    else:
        raise AttributeError("The parametric function must have exactly 2 or "
                             "3 outputs.")

    _set_misc(fig, ax, title, grid, equal_aspect)
    _show_or_return(ax, show)


def _parametric2d(x: np.ndarray, y: np.ndarray, color: str):
    """One input, two outputs. Intended to be called by parametric, rather than directly."""
    fig, ax = plt.subplots()
    ax.plot(x, y, color=color)

    return fig, ax


def _parametric3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, color: str):
    """One input, three outputs. Intended to be called by parametric, rather
    than directly."""
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z, color=color)

    return fig, ax


def _two_in_helper(f: Callable[[float, float], float], x_min: float,
                   x_max: float, y_min: float, y_max: float,
                   resolution: int) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Set up a mes grid for contour and evaluate the function for surface plots."""
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    x, y = np.meshgrid(x, y)

    return x, y, f(x, y)


def contour(f: Callable[[float, float], float], x_min: float, x_max: float,
            y_min: float=None, y_max: float=None, resolution=1e3,
            title: str=None, grid=True, show=True, equal_aspect=False,
            style: str=DEFAULT_STYLE) -> None:
    """Two inputs, one output."""
    if not y_min:
        y_min = x_min
    if not y_max:
        y_max = x_max

    x, y, z = _two_in_helper(f, x_min, x_max, y_min, y_max, resolution)

    # Style seems to require a reset, or some properties from previous styles stick.
    plt.style.use('classic')
    plt.style.use(style)  # style must be set before setting fig, ax.

    fig, ax = plt.subplots()
    ax.contour(x, y, z)

    _set_misc(fig, ax, title, grid, equal_aspect)
    _show_or_return(ax, show)


def surface(f: Callable[[float, float], float], x_min: float, x_max: float,
            y_min: float=None, y_max: float=None, title: str=None, show=True,
            equal_aspect=False, resolution=1e2, style: str=DEFAULT_STYLE) -> None:
    """Two inputs, one output."""
    if not y_min:
        y_min = x_min
    if not y_max:
        y_max = x_max

    x, y, z = _two_in_helper(f, x_min, x_max, y_min, y_max, resolution)

    # Style seems to require a reset, or some properties from previous styles stick.
    plt.style.use('classic')
    plt.style.use(style)  # style must be set before setting fig, ax.

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.coolwarm)

    _set_misc(fig, ax, title, False, equal_aspect)
    _show_or_return(ax, show)


def vector(f: Callable[[float, float], Tuple[float, float]], x_min: float,
           x_max: float, y_min: float=None, y_max: float=None, grid=True,
           title: str=None, show=True, equal_aspect=False, stream=False,
           resolution: int=15, style: str=DEFAULT_STYLE) -> None:
    """Two inputs, two outputs. 2D vector plot. steam=True sets a streamplot with curved arrows
     instead of a traditionl vector plot."""
    if not y_min:
        y_min = x_min
    if not y_max:
        y_max = x_max

    x, y, (i, j) = _two_in_helper(f, x_min, x_max, y_min, y_max, resolution)
    vec_len = (i**2 + j**2)**.5  # For color coding

    # Style seems to require a reset, or some properties from previous styles stick.
    plt.style.use('classic')
    plt.style.use(style)  # style must be set before setting fix, ax.
    fig, ax = plt.subplots()

    if stream:
        ax.streamplot(x, y, i, j, color=vec_len, cmap=cm.inferno)
    else:
        ax.quiver(x, y, i, j, vec_len, cmap=cm.inferno)

    _set_misc(fig, ax, title, grid, equal_aspect)
    _show_or_return(ax, show)


def auto(f, x_min, x_max) -> None:
    pass

