import math

from statistical_hypothesis_testing import normal_two_sided_bounds

# say, we observe 525 heads outof 1,000 flips, then we estimate p equals 0.525
p_hat = 525 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000)

lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu, sigma)
print(lower_bound, upper_bound)

# if instead we'd seen 540 heads, then we'd have:
p_hat = 540 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000)

lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu, sigma)
print(lower_bound, upper_bound)
# Here, “fair coin” doesn’t lie in the confidence interval. (The “fair coin”
# hypothesis doesn’t pass a test that you’d expect it to pass 95% of the time
# if it were true.)