from typing import List
import random

from scratch.hypothesis_and_inference.statistical_hypothesis_testing import normal_two_sided_bounds, normal_approximation_to_binomial

def run_experiment() -> List[bool]:
    """
    Filps a fair coin 1000 times, True = heads, False = tails.
    """
    return [random.random() < 0.5 for _ in range(1000)]

def reject_fairness(experiment: List[bool]) -> bool:
    """
    Using the 5% sigificant levels.
    """

    num_heads = len([flip for flip in experiment if flip])
    p = 0.5
    mu, sigma = normal_approximation_to_binomial(len(experiment), p)
    lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu, sigma)
    return num_heads < lower_bound or num_heads > upper_bound

random.seed(0)
experiments = [run_experiment() for _ in range(1000)]
num_rejections = len([experiment
                      for experiment in experiments
                      if reject_fairness(experiment)])

print(num_rejections)

# What this means is that if you’re setting out to find “significant” results,
# you usually can. Test enough hypotheses against your dataset, and one of them
# will almost certainly appear significant. Remove the right outliers, and you
# can probably get your p-value below 0.05. (We did something vaguely similar
# in “Correlation”; did you notice?)

# This is sometimes called p-hacking and is in some ways a consequence of the
# “inference from p-values framework.” A good article criticizing this approach
# is “The Earth Is Round”, by Jacob Cohen.

# If you want to do good science, you should determine your hypotheses before
# looking at the data, you should clean your data without the hypotheses in mind,
# and you should keep in mind that p-values are not substitutes for common sense. 
