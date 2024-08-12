from typing import NamedTuple, Union, List, Callable, Tuple

import matplotlib.pyplot as plt

from scratch.linear_algebra.vectors import Vector, distance, vector_mean
from scratch.clustering.example_meetups import inputs

# An alternative approach to clustering is to “grow” clusters from the bottom up. 
# We can do this in the following way:
    # 1. Make each input its own cluster of one.
    # 2. As long as there are multiple clusters remaining, find the two closest 
    #    clusters and merge them.

# At the end, we’ll have one giant cluster containing all the inputs. If we keep 
# track of the merge order, we can re-create any number of clusters by unmerging. 
# For example, if we want three clusters, we can just undo the last two merges.”

# We’ll use a really simple representation of clusters.  Our values will live in 
# leaf clusters, which we will represent as NamedTuples:

class Leaf(NamedTuple):
    value: Vector

# We’ll use these to grow merged clusters, which we will also represent as 
# NamedTuples:

class Merged(NamedTuple):
    children: tuple
    order: int


Cluster = Union[Leaf, Merged]

# We’ll talk about merge order in a bit, but first let’s create a helper function
# that recursively returns all the values contained in a (possibly merged) cluster:

def get_values(cluster: Cluster) -> List[Vector]:
    if isinstance(cluster, Leaf):
        return [cluster.value]
    else:
        return [value
                for child in cluster.children
                for value in get_values(child)]


# In order to merge the closest clusters, we need some notion of the distance 
# between clusters.  We’ll use the minimum distance between elements of the two 
# clusters, which merges the two clusters that are closest to touching (but will 
# sometimes produce large chain-like clusters that aren’t very tight). If we 
# wanted tight spherical clusters, we might use the maximum distance instead, 
# as it merges the two clusters that fit in the smallest ball. Both are common 
# choices, as is the average distance:

def cluster_distance(cluster1: Cluster,
                     cluster2: Cluster,
                     distance_agg: Callable=min) -> float:
    """
    compute all the pairwise distances between cluster1 and cluster2 and apply 
    the aggregation function _distance_agg_ to the resulting list
    """

    return distance_agg([distance(v1, v2)
                         for v1 in get_values(cluster1)
                         for v2 in get_values(cluster2)])

# We’ll use the merge order slot to track the order in which we did the merging. 
# Smaller numbers will represent later merges.  This means when we want to unmerge 
# clusters, we do so from lowest merge order to highest. Since Leaf clusters were 
# never merged, we’ll assign them infinity, the highest possible value. And since 
# they don’t have an .order property, we’ll create a helper function:

def get_merge_order(cluster: Cluster) -> float:
    if isinstance(cluster, Leaf):
        return float("inf")
    else:
        return cluster.order

# Similarly, since Leaf clusters don’t have children, we’ll create and add a 
# helper function for that:

def get_children(cluster: Cluster):
    if isinstance(cluster, Leaf):
        return TypeError("Leaf has no children")
    else:
        return cluster.children

# Now we’re ready to create the clustering algorithm:

def bottom_up_cluster(inputs: List[Vector],
                      distance_agg: Callable=min) -> Cluster:
    # start with all leave
    clusters = [Leaf(input) for input in inputs]

    def pair_distance(pair: Tuple[Cluster, Cluster]) -> float:
        c1, c2 = pair
        return cluster_distance(c1, c2, distance_agg)

    # as long as we have more than one cluster left...
    while len(clusters) != 1:
        # find the two closest clusters
        c1, c2 = min(((cluster1, cluster2)
                      for i, cluster1 in enumerate(clusters)
                      for cluster2 in clusters[i + 1:]),
                      key=pair_distance)

        # remove them from the list of cluster
        clusters = [cluster for cluster in clusters if cluster not in [c1, c2]]

        # merge them, using merge_order = # of clusters left
        merged_cluster = Merged((c1, c2), len(clusters))

        # and add their merge
        clusters.append(merged_cluster)

    return clusters[0]

# Its use is very simple:

base_cluster = bottom_up_cluster(inputs, min)

# Generally, though, we don’t want to be squinting at nasty text representations 
# like this. Instead, let’s write a function that generates any number of clusters 
# by performing the appropriate number of unmerges:

def generate_clusters(base_cluster: Cluster, num_clusters: int) -> List[Cluster]:
    # start with a list with just the base cluster
    clusters = [base_cluster]
    # as long as we don't have enough clusters yet...
    while len(clusters) < num_clusters:
        # choose the last-merged of our clusters
        next_cluster = min(clusters, key=get_merge_order)
        # remove it from the list
        clusters = [cluster for cluster in clusters if cluster != next_cluster]
        # and add its children to to list
        clusters.extend(get_children(next_cluster))

    # once we have enough clusters...
    return clusters

# So, for example, if we want to generate three clusters, we can just do:
three_clusters = [get_values(cluster)
                  for cluster in generate_clusters(base_cluster, 3)]

# which we can easily plot:

for i, cluster, marker, color in zip([1, 2, 3],
                                     three_clusters,
                                     ['D','o','*'],
                                     ['r','g','b']):
    xs, ys = zip(*cluster)  # magic unzipping trick
    plt.scatter(xs, ys, color=color, marker=marker)

    # put a number at the mean of the cluster
    x, y = vector_mean(cluster)
    plt.plot(x, y, marker='$' + str(i) + '$', color='black')

plt.title("User Locations -- 3 Bottom-Up Clusters, Min")
plt.xlabel("blocks east of city center")
plt.ylabel("blocks north of city center")
plt.show()
