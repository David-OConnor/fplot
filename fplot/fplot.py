from functools import wraps
from typing import Callable, Tuple

from matplotlib import cm, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Axes3d's required for 3d plots,
# even though it's not specifically required.
import numpy as np

#from numba import jit


# Todo broken atm.
def show2(f):
    @wraps(f)
    def inner(*args, **kwargs):
        print(args)
        print(kwargs)
        if kwargs['show']:
            plt.show()
        else:
            return f(*args, **kwargs)
    return inner



# @jit
# def ols(x, y):
#     """Simple OLS for two data sets."""
#     M = x.size
#
#     x_sum = 0.
#     y_sum = 0.
#     x_sq_sum = 0.
#     x_y_sum = 0.
#
#     for i in range(M):
#         x_sum += x[i]
#         y_sum += y[i]
#         x_sq_sum += x[i] ** 2
#         x_y_sum += x[i] * y[i]
#
#     slope = (M * x_y_sum - x_sum * y_sum) / (M * x_sq_sum - x_sum**2)
#     intercept = (y_sum - slope * x_sum) / M
#
#     return slope, intercept
#
#
# def test_linear(f):
#         # Test if linear
#         test_x = np.random.random(3)
#         test_y = f(test_x)
#         slope, intercept = ols(test_x, test_y)
#
#         if all(test_x * slope + intercept == test_y):
#             return True
#         return False

def test_linear(f):
    # Test if the function is linear; f(x) should vary proportionally with
    # x.
    x = np.arange(1, 5)  # todo this will fail for undefined vals.
    print(f(x))
    # Test if the diffs are all the same; they should be as long as x is
    # equally-spaced.
    diffs = np.diff(f(x))

    if all(diffs[1:] == diffs[:-1]):
        return True
    return False


def find_cat(f):
    """Find a function's category"""
    if test_linear(f):
        return 'linear'


def estimate_bounds(f):
    """Estimate reasonable x and y display bounds for a function"""

    if find_cat(f) == 'linear':
        y_int = f(0)
        return -5, 5

    else:
        x_test = np.logspace(-10, 10, 40)
        y_test = f(x_test)
        diffs = np.diff(y_test)
        # diffs2 = np.diff(diffs)
        # for x, diff in zip(x_test, diffs):
            # print(x, diff)
        # print(diffs2)

        # todo use slope instead of just y diff (ie take x diff into account)
        std = np.std(diffs)

        return 0, 10


def _set_grid(ax):
    ax.grid(True)

    ax.axhline(y=0, linewidth=1.5, color='k')
    ax.axvline(x=0, linewidth=1.5, color='k')
    return ax


def _set_misc(fig, ax, title, grid, equal_aspect):
    if grid:
        _set_grid(ax)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
        top_margin = .95
    else:
        top_margin = 1

    if equal_aspect:
        ax.set_aspect('equal')

    plt.tight_layout(rect=(0, 0, 1, top_margin))

    return fig, ax


def _show_or_return(ax, show):
    if show:
        plt.show()
    else:
        return ax


def plot(f: Callable[[float], float], x_min: float, x_max: float,
         title: str=None, grid=True, show=True, equal_aspect=False) -> None:
    """One input, one output."""
    resolution = 1e5

    x = np.linspace(x_min, x_max, resolution)
    y = f(x)

    fig, ax = plt.subplots()

    ax.plot(x, y)
    if equal_aspect:
        ax.set_aspect('equal')

        _set_misc(fig, ax, title, grid, equal_aspect)

    _show_or_return(ax, show)


def parametric(f: Callable[[float], Tuple[float, float]], t_min: float,
               t_max: float, title: str=None, grid=True, show=True,
               equal_aspect=False)-> None:
    """One input, two outputs (2d plot), three outputs (3d plot)."""
    resolution = 1e5

    t = np.linspace(t_min, t_max, resolution)
    outputs = f(t)

    if len(outputs) == 2:
        fig, ax = _parametric2d(*outputs)
    elif len(outputs) == 3:
        fig, ax = _parametric3d(*outputs)
        grid = False  # The grid and axes doesn't work the same way on mpl's 3d plots.
    else:
        raise AttributeError("The parametric function must have exactly 2 or "
                             "3 outputs.")

    _set_misc(fig, ax, title, grid, equal_aspect)
    _show_or_return(ax, show)


def _parametric2d(x: np.ndarray, y: np.ndarray):
    fig, ax = plt.subplots()
    ax.plot(x, y)

    return fig, ax


def _parametric3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, equal_aspect=False):
    """One input, three outputs. Intended to be called by parametric, rather than directly."""
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z)

    return fig, ax


def _two_in_one_out_helper(f: Callable[[float, float], float], x_min: float,
                           x_max: float, resolution: int) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Set up a mes grid for contour and evaluate the function for surface plots."""
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(x_min, x_max, resolution)

    x, y = np.meshgrid(x, y)
    z = f(x, y)

    return x, y, z


def contour(f: Callable[[float, float], float], x_min: float, x_max: float,
            title: str=None, grid=True, show=True, equal_aspect=False) -> None:
    """Two inputs, one output."""
    resolution = 1e3

    x, y, z = _two_in_one_out_helper(f, x_min, x_max, resolution)

    fig, ax = plt.subplots()
    ax.contour(x, y, z)

    _set_misc(fig, ax, title, grid, equal_aspect)
    _show_or_return(ax, show)


def surface(f: Callable[[float, float], float], x_min: float, x_max: float,
            title: str=None, show=True,
            equal_aspect=False) -> None:
    """Two inputs, one output."""
    resolution = 1e2

    x, y, z = _two_in_one_out_helper(f, x_min, x_max, resolution)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.coolwarm)

    _set_misc(fig, ax, title, False, equal_aspect)
    _show_or_return(ax, show)
