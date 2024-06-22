from typing import List, Dict
from collections import Counter
import math
import random

import matplotlib.pyplot as plt

from scratch.probability.the_normal_distribution import inverse_normal_cdf
from scratch.statistics.correlation import correlation
from scratch.linear_algebra.matrices import Matric, Vector, make_matric


# Exploring One-Dimensional Data
# An obvious first step is compute a few summary statistics. You'd like to know 
# how many data points you have, the smallest, the largest, the mean, and the 
# standard deviation. But even these don't necessarily give you a great 
# understanding. A good next step is to create a historgram, in which you group 
# your data into discrete buckets and count how many points fall into each bucket.

def bucketize(point: float, bucket_size: float) -> float:
    """
    Floor the point to the next lower multiple of bucket_size.
    """

    return bucket_size * math.floor(point / bucket_size)

def make_histogram(points: List[float], bucket_size: float) -> Dict[float, int]:
    """
    Buckets the points and counts how many in each bucket.
    """

    return Counter(bucketize(point, bucket_size) for point in points)

def plot_histogram(points: List[float], bucket_size: float, title: str="") -> None:
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)
    plt.show()

random.seed(0)

# uniform between -100 and 100
uniform = [200 * random.random() - 100 for _ in range(10_000)]
# normal distribution with mean 0, standard deviation 57
normal = [57 * inverse_normal_cdf(random.random()) for _ in range(10_000)]

# Both have means close to 0 and standard deviation close to 58. However, they 
# have very different distribution.
plot_histogram(uniform, 10, "Uniform Histogram")
plot_histogram(normal, 10, "Normal Histogram")


# Two Dimensions
# Of course you'd want to understand each dimension individually. But you probably 
# also want to scatter the data.

def random_normal() -> float:
    """
    Returns a random draw from a standard normal distribution.
    """

    return inverse_normal_cdf(random.random())

xs = [random_normal() for _ in range(1_000)]
ys1 = [x + random_normal() / 2 for x in xs]
ys2 = [-x + random_normal() / 2 for x in xs]

plt.scatter(xs, ys1, marker=".", color="black", label="ys1")
plt.scatter(xs, ys2, marker=".", color="gray", label="ys2")
plt.xlabel("xs")
plt.ylabel("ys")
plt.title("Very Different Joint Distribution")
plt.show()

print(correlation(xs, ys1))
print(correlation(xs, ys2))


# Many Dimensions
# With many dimensions, you'd like to know how all the dimensions relate to one 
# another. A simple approach is to look at the correlation matrix, in which the 
# entry in row i and column j is the correlation between ith dimension and the 
# jth dimension of the data.

def correlation_matric(data: List[Vector]) -> Matric:
    """
    Returns the len(data) * len(data) matrix whose (i, j)-th entry is the
    correlation betwwen data[i] and data[j].
    """

    def correlation_ij(i: int, j: int) -> float:
        return correlation(data[i], data[j])
    
    return make_matric(len(data), len(data), correlation_ij)

# A more visual approach (if you don't have too many dimensions) is to make a 
# scatter plot matric showing all the pairwise scatterplots. 

# Just some random data to show off correlation scatterplots
num_points = 100

def random_row() -> List[float]:
    row = [0.0, 0, 0, 0]
    row[0] = random_normal()
    row[1] = -5 * row[0] + random_normal()
    row[2] = row[0] + row[1] + 5 * random_normal()
    row[3] = 6 if row[2] > -2 else 0
    return row
# each row has 4 points, but really we want the columns
corr_rows = [random_row() for _ in range(num_points)]
corr_data = [list(col) for col in zip(*corr_rows)]

num_vectors = len(corr_data)
fig, ax = plt.subplots(num_vectors, num_vectors)
for i in range(num_vectors):
    for j in range(num_vectors):
        if i != j:
            ax[i][j].scatter(corr_data[j], corr_data[i])

ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
ax[0][0].set_xlim(ax[0][1].get_ylim())

plt.show()
