from typing import List
import itertools
import random

import tqdm

from scratch.linear_algebra.vectors import Vector, squared_distance, vector_mean


# One of the simplest clustering methods is k-means, in which the number of 
# clusters k is chosen in advance, after which the goal is to partition the 
# inputs into sets S1, ... , Sk  in a way that minimizes the total sum of 
# squared distances from each point to the mean of its assigned cluster.

# There are a lot of ways to assign n points to k clusters, which means that 
# finding an optimal clustering is a very hard problem. We’ll settle for an 
# iterative algorithm that usually finds a good clustering:
    # 1. Start with a set of k-means, which are points in d-dimensional space.
    # 2. Assign each point to the mean to which it is closest.
    # 3. If no point’s assignment has changed, stop and keep the clusters.
    # 4. If some point’s assignment has changed, recompute the means and return to step 2.

def num_differences(v1: Vector, v2: Vector) -> int:
    assert len(v1) == len(v2)
    return len([1 for x1, x2 in zip(v1, v2) if x1 != x2])

assert num_differences([1, 2, 3], [2, 1, 3]) == 2
assert num_differences([1, 2], [1, 2]) == 0

# We also need a function that, given some vectors and their assignments to 
# clusters, computes the means of the clusters. It may be the case that some 
# cluster has no points assigned to it. We can’t take the mean of an empty 
# collection, so in that case we’ll just randomly pick one of the points to serve 
# as the “mean” of that cluster:

def cluster_mean(k: int, inputs: List[Vector], assignments: List[int]) -> List[Vector]:
    clusters = [[] for _ in range(k)]
    for input, assignment in zip(inputs, assignments):
        clusters[assignment].append(input)

    return [vector_mean(cluster) if cluster else random.choice(inputs)
            for cluster in clusters]

# And now we’re ready to code up our clusterer. As usual, we’ll use tqdm to 
# track our progress, but here we don’t know how many iterations it will take, 
# so we then use itertools.count, which creates an infinite iterable, and we’ll 
# return out of it when we’re done:

class KMeans:
    
    def __init__(self, k: int) -> None:
        self.k = k
        self.means = None

    def classify(self, input: Vector) -> int:
        """
        Return the index of the cluster closest to the input.
        """

        return min(range(self.k),
                   key=lambda i: squared_distance(input, self.means[i]))

    def train(self, inputs: List[Vector]) -> None:
        # start with random assignment
        assignments = [random.randrange(self.k) for _ in inputs]

        with tqdm.tqdm(itertools.count()) as t:
            for _ in t:
                # compute means and find new assignments
                self.means = cluster_mean(self.k, inputs, assignments)
                new_assignments = [self.classify(input) for input in inputs]

                # check how many assignments changed and if we're done
                num_changed = num_differences(assignments, new_assignments)
                if not num_changed:
                    return

                # otherwise keep the new assignment, and compute new means
                assignments = new_assignments
                self.means = cluster_mean(self.k, inputs, assignments)
                t.set_description(f"changed: {num_changed} / {len(inputs)}")
