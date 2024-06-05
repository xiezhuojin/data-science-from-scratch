from statistical_hypothesis_testing import \
    normal_probability_above, normal_probability_below, normal_approximation_to_binomial

def two_sided_p_value(x: float, mu: float=0, sigma: float=1) -> float:
    """
    How likely are we to see a value at least as extreme as x (in either direction)
    if our values are from an N(mu, sigma)?
    """

    if x >= mu:
        return 2 * normal_probability_above(x, mu, sigma)
    if x < mu:
        return 2 * normal_probability_below(x, mu, sigma)

# if we were to see 530 heads, we would compute:
mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
p = two_sided_p_value(529.5, mu_0, sigma_0)
print(p)

import random

extreme_value_count = 0
for _ in range(1000):
    num_heads = sum(1 if random.random() < 0.5 else 0 for _ in range(1000))
    if num_heads >= 530 or num_heads <= 470:
        extreme_value_count += 1

# assert 59 < extreme_value_count < 65, f"{extreme_value_count}"

p_1 = two_sided_p_value(531.5, mu_0, sigma_0)
print(p_1)