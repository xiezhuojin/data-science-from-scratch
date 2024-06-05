import numpy as np
import matplotlib.pyplot as plt

from scratch.linear_algebra.vectors import Vector, dot

# Suppose we have some function f that takes as input a vector of real numbers and outputs a single
# real number.  One simple such function is:

def sum_of_squares(v: Vector):
    """
    Computes the sum of squared elements in v.
    """
    return dot(v, v)

ys = xs = np.linspace(-6, 6, 1000)
zs = np.array([[sum_of_squares([x, y]) for y in ys] for x in xs])
xs, ys = np.meshgrid(xs, ys)

ax = plt.figure().add_subplot(111, projection="3d")
ax.plot_surface(xs, ys, zs)
plt.show()