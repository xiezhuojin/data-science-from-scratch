import random
from typing import List

import matplotlib.pyplot as plt

from scratch.linear_algebra.vectors import Vector, distance


# The k-nearest neighbors algorithm runs into trouble in higher dimensions thanks 
# to the "curse of dimensionality", which boils down to the fact that high-dimensional 
# spaces are vast. Points in high-dimensional spaces tend not to be close to one 
# another at all. One way to see this is by randomly generating pairs of points 
# in the d-dimensional "unit cube" in a variety of dimensions, and calculating 
# the distances between them.

def random_point(dim: int) -> Vector:
    return [random.random() for _ in range(dim)]

def random_distance(dim: int, num_pairs: int) -> List[float]:
    return [distance(random_point(dim), random_point(dim)) for _ in range(num_pairs)]

# For every dimension from 1 to 100, we'll compute 10,000 distnaces and use those 
# to compute the average distance between points and the minimun distance between 
# points in each dimension.

dimensions = range(1, 101)

avg_distances = []
min_distances = []

for i in dimensions:
    distances = random_distance(i, 10_000)
    avg_distances.append(sum(distances) / 10_000)
    min_distances.append(min(distances))

# As the number of dimensions increases, the average distance between points increses. 
# But what's more problematic is the ratio between the closest distance and the 
# average distance.

min_avg_ratio = [min_distance / avg_distance
                 for min_distance, avg_distance in zip(min_distances, avg_distances)]

plt.plot([x for x in dimensions], avg_distances, label="average distance")
plt.plot([x for x in dimensions], min_distances, label="minimum distance")
plt.xlabel("# of dimensions")
plt.title("10,000 Random Distances")
plt.legend()
plt.show()

plt.plot([x for x in dimensions], min_avg_ratio)
plt.xlabel("# of dimensions")
plt.title("Minimum Distance / Average Distance")
plt.show()

# In low-dimensional datasets, the closest points tend to be much closer than 
# average. But two points are close only if they're close in every dimension, 
# and every extra dimension - even if just noise - is another opportunity for each 
# point to be farther away from every other point. When you have a lot of 
# dimensions, it's likely that the closest points aren't much closer than average, 
# so two points being close doesn't mean very much (unless there's a lot of 
# structure in your data that makes it behave as if it were much lower-dimensional).

# A different way of thinking about the problem involves the sparsity of higher-dimensional 
# spaces.

# If you pick 50 random numbers between 0 and 1, you'll probably get a pretty 
# good sample of the unit interval.
random_points = [random.random() for _ in range(50)]
plt.bar(random_points, [1 for _ in range(50)], width=0.001)
plt.show()

# If you pick 50 random points in the unit square, you'll get less coverage.
random_points = [(random.random(), random.random()) for _ in range(50)]
xs = [point[0] for point in random_points]
ys = [point[1] for point in random_points]
plt.scatter(xs, ys)
plt.show()

# So if you're trying to use nearest neighbors in higher dimensions, it's probably 
# a good idea to do some kind of dimensionality reduction first.