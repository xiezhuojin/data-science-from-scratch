import math
from scipy.stats import norm
import matplotlib.pyplot as plt


# The normal distribution is the classic bell curve–shaped distribution and is
# completely determined by two parameters: its mean μ (mu) and its standard
# deviation σ (sigma). The mean indicates where the bell is centered, and the
# standard deviation how “wide” it is.

SQRT_TWO_PI = math.sqrt(2 * math.pi)

def normal_pdf(x: float, mu: float=0, sigma: float=1) -> float:
    return (math.exp(-(x - mu) ** 2 / 2 / sigma ** 2) / (SQRT_TWO_PI * sigma))

# same as above
norm()

xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs, norm.pdf(xs), "-", label="mu=0,sigma=1")
plt.plot(xs, norm.pdf(xs, scale=2), "--", label="mu=0,sigma=2")
plt.plot(xs, norm.pdf(xs, scale=0.5), ":", label="mu=0,sigma=0.5")
plt.plot(xs, norm.pdf(xs, loc=-1), "-.", label="mu=-1,sigma=1")
plt.legend()
plt.title("Various Normal pdfs")
plt.show()

def normal_cdf(x: float, mu: float=0, sigma: float=1) -> float:
   return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

# same as above
norm.cdf

plt.plot(xs, norm.cdf(xs), "-", label="mu=0,sigma=1")
plt.plot(xs, norm.cdf(xs, scale=2), "--", label="mu=0,sigma=2")
plt.plot(xs, norm.cdf(xs, scale=0.5), ":", label="mu=0,sigma=0.5")
plt.plot(xs, norm.cdf(xs, loc=-1), "-.", label="mu=-1,sigma=1")
plt.legend(loc=4)
plt.title("Various Normal cdfs")
plt.show()

def inverse_normal_cdf(p: float, mu: float=0, sigma: float=1, tolerance: float=0.00001) -> float:
    """
    Find approximate inverse using binary search.
    """

    # if not standard, compute standard and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z = -10.0
    hi_z = 10.0
    while hi_z - low_z > tolerance:
        mid_z = (hi_z + low_z) / 2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:
            low_z = mid_z
        else:
            hi_z = mid_z
    return mid_z

# same as above
norm.ppf