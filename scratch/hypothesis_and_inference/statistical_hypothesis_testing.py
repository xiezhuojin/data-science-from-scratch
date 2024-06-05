from typing import Tuple
import math

from scipy.stats import norm


def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    """
    Return mu and sigma corresponding to a Binomial(n, p).
    """

    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

def normal_probability_below(lo: float, mu: float=0, sigma: float=1) -> float:
    """
    The probability that an N(mu, sigma) is leass than lo.
    """

    return norm.cdf(lo, mu, sigma)

# It's above the threshold if it's not below the threshold
def normal_probability_above(lo: float, mu: float=0, sigma: float=1) -> float:
    """
    The probability that an N(mu, sigma) is greater than lo.
    """

    return 1 - norm.cdf(lo, mu, sigma)

# It's between if it's less than hi, but not less than lo
def normal_probability_between(lo: float, hi: float, mu: float=0, sigma: float=1) -> float:
    """
    The probability than an N(mu, sigma) is between lo and hi.
    """

    return norm.cdf(hi, mu, sigma) - norm.cdf(lo, mu, sigma)

# It's outside if it's not between
def normal_probability_outside(lo: float, hi: float, mu: float=0, sigma: float=1) -> float:
    """
    The probability that an N(mu, sigma) is not between lo to hi.
    """

    return 1 - normal_probability_between(lo, hi, mu, sigma)

def normal_upper_bound(probability: float, mu: float=0, sigma: float=1) -> float:
    """
    Returns the z for which P(Z <= z) = probability.
    """

    return norm.ppf(probability, mu, sigma)

def normal_lower_bound(probability: float, mu: float=0, sigma: float=1) -> float:
    """
    Return the z for which P(Z >= z) = probability.
    """

    return norm.ppf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability: float, mu: float=0, sigma: float=1) -> Tuple[float, float]:
    """
    Return the symmetric (about the mean) bounds that contains the specified probability.
    """

    tail_probability = (1 - probability) / 2
    # upper bound should have tail_probability above it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)
    # lower bound shold have tail_probability below it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

# In particular, let’s say that we choose to flip the coin n = 1,000 times. If
# our hypothesis of fairness is true, X should be distributed approximately
# normally with mean 500 and standard deviation 15.8

mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
print(mu_0, sigma_0)

# We need to make a decision about significance—how willing we are to make a type
# 1 error (“false positive”), in which we reject H0 even though it’s true. 
# For reasons lost to the annals of history, this willingness is often set at 5%
# or 1%. Let’s choose 5%.

# Consider the test that rejects H0  if X falls outside the bounds given by:
# (469, 531)

lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)
print(lower_bound, upper_bound)

# We are also often interested in the power of a test, which is the probability
# of not making a type 2 error  (“false negative”), in which we fail to reject 
# H0 even though it’s false.  In order to measure this, we have to specify what
# exactly H0 being false means.  (Knowing merely that p is not 0.5 doesn’t give
# us a ton of information about the distribution of X.) In particular, let’s
# check what happens if p is really 0.55, so that the coin is slightly biased
# toward heads.

# 95% bounds based on assumption p is 0.5
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)
print(lo, hi)

# actual mu and sigma base on p = 0.55
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)

# a type error means we fail to reject the null hypothesis, which iwll happen when
# X is still in our original interval
type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probability
print(power)