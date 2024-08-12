from typing import List

import matplotlib.pyplot as plt

from scratch.linear_algebra.vectors import squared_distance
from scratch.linear_algebra.vectors import Vector, squared_distance
from scratch.clustering.the_model import KMeans
from scratch.clustering.example_meetups import inputs

# In the previous example, the choice of k was driven by factors outside of our 
# control.  In general, this won’t be the case.  There are various ways to choose 
# a k. One that’s reasonably easy to understand involves plotting the sum of 
# squared errors (between each point and the mean of its cluster) as a function 
# of k and looking at where the graph “bends”:

def squared_clustering_errors(inputs: List[Vector], k: int) -> float:
    """
    Finds the total squared error from k-means clustering the inputs.
    """

    cluster = KMeans(k)
    cluster.train(inputs)
    means = cluster.means
    assignments = [cluster.classify(input) for input in inputs]

    return sum([squared_distance(input, means[assignment])
                for input, assignment in zip(inputs, assignments)])
    
# now plot from 1 up to len(inputs) clusters
ks = range(1, len(inputs) + 1)
errors = [squared_clustering_errors(inputs, k) for k in ks]

plt.plot(ks, errors)
plt.xticks(ks)
plt.xlabel("k")
plt.ylabel("total squared error")
plt.title("Total Error vs. # of Clusters")
plt.show()

# This method agrees with our original eyeballing that three is the “right” 
# number of clusters.
