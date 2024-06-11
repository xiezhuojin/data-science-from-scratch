from typing import Callable, List

from scipy.misc import derivative
import matplotlib.pyplot as plt


def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    return (f(x + h) - f(x)) / h

def square(x: float) -> float:
    return x ** 2

def square_derivative(x: float) -> float:
    return 2 * x

xs = range(-10, 11)
actuals = [square_derivative(x) for x in xs]
estimates = [difference_quotient(square, x, 0.001) for x in xs]

# plot to show they're basically the same
plt.title("Actual Derivatives vs. Estimates")
plt.plot(xs, actuals, "rx", label="actual")
plt.plot(xs, estimates, "b+", label="Estimate")
plt.legend()
plt.show()

def partial_difference_quotient(f: Callable[[List[float]], float],
                                v: List[float],
                                i: int,
                                h: float) -> float:
    """
    Return the i-th partial difference quotient of f at v.
    """

    w = [v_j + h if j == 1 else v_j for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h

# same as above
derivative

def estimate_gradient(f: Callable[[List[float]], float],
                      v: List[float],
                      h: float=0.000_1):
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]

plt.title("Actual Derivatives vs. Estimates using derivative")
estimates = [derivative(square, x, 0.000_1) for x in xs]
plt.plot(xs, actuals, "rx", label="actual")
plt.plot(xs, estimates, "b+", label="Estimate")
plt.legend()
plt.show()