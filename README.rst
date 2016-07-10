Fplot: Function plots with less code
====================================
A thin wrapper for matplotlib

Fplot's goal is to 


Included functions
------------------

 - plot: One input, one output.
 - parametric: One input, two or three outputs. If three, use a 3d graph.
 - contour: Two inputs, one output.
 - surface: Two inputs, one output.
 - vector: Two inputs, two outputs.



Installation
------------

.. code-block:: python

    pip install fplot


Basic documentation
-------------------



.. code-block:: python

    f = lambda x: x**2 + 3
    fplot.plot(f, -10, 10)
    # Insert image


