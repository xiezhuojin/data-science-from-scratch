import math
from typing import List
from collections import Counter

import matplotlib.pyplot as plt

from scratch.linear_algebra.vectors import sum_of_squares


num_friends = [100.0,49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

friend_counts = Counter(num_friends)
xs = range(101)
ys = [friend_counts[x] for x in xs]
plt.bar(xs, ys)
plt.axis([0, 101, 0, 25])
plt.title("Histogram of Friend Counts")
plt.xlabel("# of friends")
plt.ylabel("# of people")
plt.show()

num_points = len(num_friends)
largest_value = max(num_friends)
smallest_value = min(num_friends)
assert num_points == 204
assert largest_value == 100
assert smallest_value == 1

# Central Tendencies

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

def _median_odd(xs: List[float]) -> float:
    """
    If len(xs) is odd, the median is the middle element.
    """

    return sorted(xs)[len(xs) // 2]

def _median_even(xs: List[float]) -> float:
    """
    If len(xs) is even, it's the average of the middle two elements.
    """

    sorted_xs = sorted(xs)
    mid = len(xs) // 2
    return (sorted_xs[mid - 1] + sorted_xs[mid]) / 2

def median(v: List[float]) -> float:
    """
    FInds the 'middle-most' value of v.
    """

    return _median_odd if len(v) % 2 == 1 else _median_even(v)

def quantile(xs: List[float], p:  float) -> float:
    """
    Returns the pth-percentile vlaue in x
    """

    p_index = int(len(xs) * p)
    return sorted(xs)[p_index]

def mode(x: List[float]) -> List[float]:
    """
    Return a list, since there might be more than one mode.
    """

    counter = Counter(x)
    return [k for k, v in counter.items() if v == counter.most_common(1)[0][1]]
    

assert median(num_friends) == 6
assert quantile(num_friends, 0.10) == 1
assert quantile(num_friends, 0.25) == 3
assert quantile(num_friends, 0.75) == 9
assert quantile(num_friends, 0.90) == 13
assert set(mode(num_friends)) == {1, 6}

# Dispersion

def data_range(xs: List[float]) -> float:
    return max(xs) - min(xs)

def de_mean(xs: List[float]) -> List[float]:
    """
    Translate xs by subtracting its mean (so the result has mean 0)
    """

    xs_mean = mean(xs)
    return [x - xs_mean for x in xs]

def variance(xs: List[float]) -> float:
    """
    Almost the average squared deviation form the mean.
    """

    assert len(xs) > 2, "variance requires at least two elements"

    return sum_of_squares(de_mean(xs)) / (len(xs) - 1)

def standard_deviation(xs: List[float]) -> float:
    """
    The standard deviation is the square root of the variance.
    """

    return math.sqrt(variance(xs))

def interquartile_range(xs: List[float]) -> float:
    """
    Returns the difference between the 75%-ile and 25%-ile.
    """

    return quantile(xs, 0.75) - quantile(xs, 0.25)

assert data_range(num_friends) == 99
assert 81.54 < variance(num_friends) < 81.55
assert 9.02 < standard_deviation(num_friends) < 9.04
assert interquartile_range(num_friends) == 6
