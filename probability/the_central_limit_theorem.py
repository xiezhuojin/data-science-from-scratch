import math
import random
import numpy
from collections import Counter
import matplotlib.pyplot as plt

from scipy.stats import norm

# One reason the normal distribution is so useful is the central limit theorem,
# which says (in essence) that a random variable defined as the average of a
# large number of independent and identically distributed random variables is
# itself approximately normally distributed.

def bernoulli_trial(p: float) -> int:
    """
    Returns 1 with probability p and 0 with probability 1-p.
    """

    return 1 if random.random() < p else 0

def binomial(n: int, p: float) -> int:
    """
    Returns the sum of n bernoulli(p) trials.
    """

    return sum(bernoulli_trial(p) for _ in range(n))

def binomial_historgram(p: float, n: int, num_points: int) -> None:
    """
    Picks points from a Binomial(n, p) and plots their histogram.
    """

    data = [binomial(n, p) for _ in range(num_points)]

    # use a bar chart to show the actual binomial samples
    historgram = Counter(data)
    plt.bar([x - 0.4 for x in historgram.keys()],
            [v / num_points for v in historgram.values()],
            0.8,
            color="0.75"
            )
    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))

    # use a line chart to show the normal approximation
    xs = numpy.arange(min(data), max(data) + 1)
    # ys = norm.cdf([x + 0.5 for x in xs], loc=mu, scale=sigma) - norm.cdf(xs - 0.5, loc=mu, scale=sigma)
    ys = norm.pdf(xs, loc=mu, scale=sigma)
    plt.plot(xs, ys)
    plt.title("Binomial Distribution vs. Normal Approximation")
    plt.show()

binomial_historgram(0.75, 100, 10_000)