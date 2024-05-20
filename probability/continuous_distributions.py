from scipy.stats import uniform
import numpy
import matplotlib.pyplot as plt

# the uniform distribution puts equal weight on all the numbers between 0 and 1.
# Because there are infinitely many numbers between 0 and 1, this means that the
# weight it assigns to individual points must necessarily be zero. For this
# reason, we represent a continuous distribution with a probability density
# function (PDF) such that the probability of seeing a value in a certain
# interval equals the integral of the density function over the interval.

def uniform_pdf(x: float) -> float:
    return 1 if 0 <= x <= 1 else 0

# just the same as above
uniform.pdf

# We will often be more interested in the cumulative distribution function (CDF),
# which gives the probability that a random variable is less than or equal to a
# certain value.

def uniform_cdf(x: float) -> float:
    """
    Returns the probability that a uniform random variable is <= x.
    """

    if x < 0:
        return 0
    elif x < 1:
        return x
    else:
        return 1
    
# just the same as above
uniform.cdf

xs = numpy.arange(-1, 2, 0.5)
ys = [uniform.cdf(x) for x in xs]

plt.plot(xs, ys)
plt.title("The uniform cdf")
plt.show()