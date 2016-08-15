Fplot: Function plots with less code
====================================
A thin wrapper for matplotlib

Fplot's goal is to provide simple syntax for plotting functions, with sensible
defaults. Matplotlib is powerful, but has awkward syntax, odd default display settings,
and requires setting up data arrays manually. Making pretty function plots requires
multiple lines of code. Fplot aims to fix this; it's especially suited for visualizing
functions when learning math.

Python 3 only. This is an early, unpolished release - I'm welcome to suggestions.


Included functions
------------------

 - plot: One input, one output.
 - parametric: One input, two or three outputs. If three, use a 3d graph.
 - contour: Two inputs, one output.
 - surface: Two inputs, one output.
 - vector: Two inputs, two outputs.
 - polar: One input (angle), two outputs (radius)


Installation
------------

.. code-block:: python

    pip install fplot


Basic documentation
-------------------

`Examples
<https://github.com/David-OConnor/fplot/blob/master/examples.ipynb/>`_


The only required arguments for fplot funcs are the function to plot, and the
min and max ranges. Example optional keyword arguments are shown. Example output
is shown in the link above.

Show a graph (1 input, 1 output)

.. code-block:: python

    f = lambda x: x**2 + 2
    fplot.plot(f, -10, 10, title='Hello world')


Show a contour plot (2 inputs, 1 output)

.. code-block:: python

    g = lambda x, y: x**2 + y**2 + 10
    fplot.contour(g, -10, 10, equal_aspect=True)


Show a surface plot (2 inputs, 1 output)

.. code-block:: python

    g = lambda x, y: x**2 + y**2 + 10
    fplot.surface(g, -10, 10)


Show a 2d parametric plot (1 input, 2 outputs)

.. code-block:: python

    h = lambda t: (np.sin(t), np.cos(t))
    fplot.parametric(h, 0, τ, equal_aspect=True, color='m')


Show a 3d parametric plot (1 input, 3 outputs)

.. code-block:: python

    i = lambda t: (np.sin(t), np.cos(t), t**2)
    fplot.parametric(i, 0, 20, color='red')


Show a 2d vector plot (2 inputs, 2 outputs)

.. code-block:: python

    f = lambda x, y: (x**2 + y, y**2 * cos(x))
    fplot.vector(f, -10, 10, stream=False)


Show a 3d vector plot (3 inputs, 3 outputs)

.. code-block:: python

    f = lambda x, y, z: (x**2, y**2, z)
    fplot.vector3d(f, -10, 10)


Show a 2d polar plot (1 input, 1 output)

.. code-block:: python

    f = lambda theta: np.sin(3*theta)
    fplot.polar(f, 0, tau, color='purple')


Optional arguments:
 - show: Defaults to True. Instantly display the plot. If False, return the axis object.
 - resolution: Controls how many points to draw, based on function input. Higher resolution
   allows more zooming, but may lower performance.
 - color: (ie line color)
 - y_min and y_max: (only for 2d input)
 - theta_min and theta_max (only for polar plots)
 - style: (ie from plt.use.style())
 - grid: defaults to True
 - equal_aspect: defaults to False
 - title: Shown at the top of the plot
 - stream: vector plot only; show a stream plot if True
 - contours: surface plot only; show contour plots along each axis if True