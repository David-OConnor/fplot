from functools import wraps
from typing import Callable, Tuple

from matplotlib import cm, pyplot as plt
# Axes3d's required for 3d plots, even though it's not specifically required.
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


DEFAULT_STYLE = 'seaborn-deep'
DEFAULT_COLORMAP = cm.viridis

# todolook into strike and rstrike for 3d plots.

τ = 2 * np.pi

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



    # If a vertical asympytote exists, set y display range to a reasonable value.
    max_, min_, median, std = np.max(y), np.min(y), np.median(y), np.std(y)

    lower_lim, upper_lim = min_, max_

    m = 4  # Number of standard deviations required to trigger an adjustment.
    dev = np.abs(y - np.median(y))
    dev_low, dev_high = median - y, y - median
    mdev = np.median(dev)

    flag = False
    if any(dev_high > np.std(dev_high) * m):
        flag = True
        upper_lim = median + 1*std
    if any(dev_low < -np.std(dev_low) * m):
        flag = True
        lower_lim = median - 1*std

    if flag:
        ax.set_ylim([lower_lim, upper_lim])
    #todo fix the above

    if equal_aspect:
        ax.set_aspect('equal')

    _set_misc(fig, ax, title, grid, equal_aspect)
    return _show_or_return(ax, show)


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
    return _show_or_return(ax, show)


def parametric_surface(f: Callable[[float, float], Tuple[float, float, float]], t_min: float,
               t_max: float, s_min: float=None, s_max: float=None, title: str=None,
               grid=True, show=True, equal_aspect=False, alpha: float=1.0,
               resolution=1e2, style: str=DEFAULT_STYLE)-> None:
    if not s_min:
        s_min = t_min
    if not s_max:
        s_max = t_max

    tau = 2*np.pi
    x_min, x_max = -tau, tau
    y_min, y_max = -tau, tau
    z_min, z_max = -10, 10

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    # z = np.linspace(z_min, z_max, resolution)
    x_mesh, y_mesh = np.meshgrid(x, y)
    r = np.sqrt(x_mesh, y_mesh)

    t = np.linspace(t_min, t_max, resolution)
    s = np.linspace(s_min, s_max, resolution)

    x, y, z = f(t, s)


    # ###
    # du = np.sqrt(np.diff(x, axis=0) ** 2 + np.diff(y, axis=0) ** 2 + np.diff(z,
    #                                                                          axis=0) ** 2)
    # dv = np.sqrt(np.diff(x, axis=1) ** 2 + np.diff(y, axis=1) ** 2 + np.diff(z,
    #                                                                          axis=1) ** 2)
    # u = np.zeros_like(x)
    # v = np.zeros_like(x)
    # u[1:, :] = np.cumsum(du, axis=0)
    # v[:, 1:] = np.cumsum(dv, axis=1)
    #
    # u /= u.max(axis=0)[None,
    #      :]  # hmm..., or maybe skip this scaling step -- may distort the result
    # v /= v.max(axis=1)[:, None]
    #
    # # construct interpolant (unstructured grid)
    # from scipy import interpolate
    # ip_surf = interpolate.CloughTocher2DInterpolator(
    #     (u.ravel(), v.ravel()),
    #     np.c_[x.ravel(), y.ravel(), z.ravel()])

    # the BivariateSpline classes might also work here, but the above is more robust

    # plot projections
    #
    # u = np.random.rand(2000)
    # v = np.random.rand(2000)
    #
    # plt.subplot(131)
    # plt.plot(ip_surf(u, v)[:, 0], ip_surf(u, v)[:, 1], '.')
    # plt.subplot(132)
    # plt.plot(ip_surf(u, v)[:, 1], ip_surf(u, v)[:, 2], '.')
    # plt.subplot(133)
    # plt.plot(ip_surf(u, v)[:, 2], ip_surf(u, v)[:, 0], '.')
    #
    # plt.show()
    # return

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z)

    ax.plot_surface(x, y, z, cmap=DEFAULT_COLORMAP, alpha=alpha)

    # ax.set_xlabel('x-axis')
    # ax.set_ylabel('y-axis')
    # ax.set_zlabel('z-axis')

    _set_misc(fig, ax, title, grid, equal_aspect)
    return _show_or_return(ax, show)


def _parametric2d(x: np.ndarray, y: np.ndarray, color: str):
    """Two outputs. Intended to be called by parametric, rather than directly."""
    fig, ax = plt.subplots()
    ax.plot(x, y, color=color)

    return fig, ax


def _parametric3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, color: str):
    """One input, three outputs. Intended to be called by parametric, rather
    than directly."""
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z)

    return fig, ax


def _two_in_one_out_helper(f: Callable[[float, float], float], x_min: float,
                           x_max: float, y_min: float, y_max: float,
                           resolution: int) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Set up a mesh grid for contour and evaluate the function for surface plots."""
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    x_mesh, y_mesh = np.meshgrid(x, y)

    return x_mesh, y_mesh, f(x_mesh, y_mesh)


def contour(f: Callable[[float, float], float], x_min: float, x_max: float,
            y_min: float=None, y_max: float=None, resolution=1e3,
            title: str=None, grid=True, show=True, equal_aspect=False,
            style: str=DEFAULT_STYLE) -> None:
    """Two inputs, one output."""
    if not y_min:
        y_min = x_min
    if not y_max:
        y_max = x_max

    x_mesh, y, z = _two_in_one_out_helper(f, x_min, x_max, y_min, y_max, resolution)

    # Style seems to require a reset, or some properties from previous styles stick.
    plt.style.use('classic')
    plt.style.use(style)  # style must be set before setting fig, ax.

    fig, ax = plt.subplots()
    ax.contour(x_mesh, y, z)

    _set_misc(fig, ax, title, grid, equal_aspect)
    return _show_or_return(ax, show)


def surface(f: Callable[[float, float], float], x_min: float, x_max: float,
            y_min: float=None, y_max: float=None, title: str=None, show=True,
            equal_aspect=False, contours=False, resolution=1e2, style: str=DEFAULT_STYLE) -> None:
    """Two inputs, one output."""
    if not y_min:
        y_min = x_min
    if not y_max:
        y_max = x_max

    x_mesh, y_mesh, z_mesh = _two_in_one_out_helper(f, x_min, x_max, y_min, y_max, resolution)

    # Style seems to require a reset, or some properties from previous styles stick.
    plt.style.use('classic')
    plt.style.use(style)  # style must be set before setting fig, ax.

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    alpha = .3 if contours else 1.0
    ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap=DEFAULT_COLORMAP, alpha=alpha)

    if contours:
        offset_dist = .2  # How far from the graph to draw the contours, as a
        # of min and max values.
        x_offset = x_min - offset_dist * (x_max - x_min)
        y_offset = y_max + offset_dist * (y_max - y_min)
        z_offset = z_mesh.min() - offset_dist * (z_mesh.max() - z_mesh.min())

        ax.contour(x_mesh, y_mesh, z_mesh, zdir='x', offset=x_offset, cmap=DEFAULT_COLORMAP)
        ax.contour(x_mesh, y_mesh, z_mesh, zdir='y', offset=y_offset, cmap=DEFAULT_COLORMAP)
        ax.contour(x_mesh, y_mesh, z_mesh, zdir='z', offset=z_offset, cmap=DEFAULT_COLORMAP)

    _set_misc(fig, ax, title, False, equal_aspect)
    return _show_or_return(ax, show)


def polar(f: Callable[[float], float], theta_min: float=0, theta_max: float=τ,
          title: str=None, color: str=None, resolution: int=1e5, show: bool=True) -> None:
    """Make a polar plot. Function input is theta, in radians; output is radius.
    0 radians corresponds to a point on the x axis, with positive y, ie right side.
    Goes counter-clockwise from there."""
    # todo more customization for xticks (ie theta ticks) ie degrees, pi, 4/8/16 divisions etc
    θ = np.linspace(theta_min, theta_max, resolution)
    r = f(θ)

    fig = plt.figure()
    ax = fig.gca(projection='polar')
    ax.plot(θ, r, color=color)

    ax.set_xticks(np.linspace(0, 15*τ/16, 16))

    ax.set_xticklabels(['0', r'$\frac{\tau}{16}$', r'$\frac{\tau}{8}$', r'$\frac{3\tau}{16}$',
          r'$\frac{\tau}{4}$', r'$\frac{5\tau}{16}$', r'$\frac{3\tau}{8}$', r'$\frac{7\tau}{16}$',
          r'$\frac{\tau}{2}$', r'$\frac{9\tau}{16}$', r'$\frac{5\tau}{8}$', r'$\frac{11\tau}{16}$',
          r'$\frac{3\tau}{4}$', r'$\frac{13\tau}{16}$', r'$\frac{7\tau}{8}$', r'$\frac{15\tau}{16}$'],
          fontsize=20)

    # Default figure size is too small.
    fig.set_size_inches(8, 8, forward=True)
    # xL = ['0', r'$\frac{\tau}{8}$', r'$\frac{\tau}{4}$',
    #       r'$\frac{3\tau}{8}$',
    #       r'$\frac{\tau}{2}$', r'$\frac{5\tau}{8}$', r'$\frac{3\tau}{4}$',
    #       r'$\frac{7\tau}{8}$']

    # xL = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$',
    #       r'$\frac{3\pi}{4}$',
    #       r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$',
    #       r'$\frac{7\pi}{4}$']
    # plt.xticks(xT, xL)

    _set_misc(fig, ax, title, False, None)
    return _show_or_return(ax, show)


def vector(f: Callable[[float, float], Tuple[float, float]], x_min: float,
           x_max: float, y_min: float=None, y_max: float=None, grid=True,
           title: str=None, show=True, equal_aspect=False, stream=False,
           resolution: int=15, style: str=DEFAULT_STYLE) -> None:
    """Two inputs, two outputs. 2D vector plot. stream=True sets a streamplot
    with curved arrows instead of a traditionl vector plot."""
    if not y_min:
        y_min = x_min
    if not y_max:
        y_max = x_max

    x, y, (i, j) = _two_in_one_out_helper(f, x_min, x_max, y_min, y_max, resolution)
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
    return _show_or_return(ax, show)


def vector3d(f: Callable[[float, float, float], Tuple[float, float, float]],
             x_min: float, x_max: float, y_min: float=None, y_max: float=None,
             z_min: float=None, z_max: float=None, title: str=None, show=True, equal_aspect=False, resolution: int=5,
             style: str=DEFAULT_STYLE) -> None:
    """Three inputs, three outputs. 3D vector plot. stream=True sets a streamplot
    with curved arrows instead of a traditionl vector plot."""
    if not y_min:
        y_min = x_min
    if not y_max:
        y_max = x_max
    if not z_min:
        z_min = x_min
    if not z_max:
        z_max = x_max

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    z = np.linspace(z_min, z_max, resolution)
    x, y, z = np.meshgrid(x, y, z)

    i, j, k = f(x, y, z)
    vec_len = (i**2 + j**2 + k**2) ** .5  # For color coding

    # Style seems to require a reset, or some properties from previous styles stick.
    plt.style.use('classic')
    plt.style.use(style)  # style must be set before setting fix, ax.

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # todo: Unsure how to support colors.
    ax.quiver(x, y, z, i, j, k, cmap=cm.inferno)

    _set_misc(fig, ax, title, False, equal_aspect)
    return _show_or_return(ax, show)

