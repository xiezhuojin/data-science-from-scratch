from typing import List
import random

import matplotlib.pyplot as plt

from scratch.linear_algebra.vectors import squared_distance
from scratch.clustering.the_model import KMeans

# To celebrate DataSciencester’s growth, your VP of User Rewards wants to 
# organize several in-person meetups for your hometown users, complete with 
# beer, pizza, and DataSciencester t-shirts.  You know the locations of all your
#  local users (Figure 20-1), and she’d like you to choose meetup locations that
#  make it convenient for everyone to attend.

inputs: List[List[float]] = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]

xs = [input[0] for input in inputs]
ys = [input[1] for input in inputs]

plt.scatter(xs, ys)
plt.xlabel("blocks east of city center")
plt.ylabel("blocks north of city center")
plt.title("User Locations")
plt.show()

# Depending on how you look at it, you probably see two or three clusters.  
# (It’s easy to do visually because the data is only two-dimensional. With more 
# dimensions, it would be a lot harder to eyeball.)

# Imagine first that she has enough budget for three meetups. You go to your 
# computer and try this:

random.seed(12)
cluster = KMeans(3)
cluster.train(inputs)
means = sorted(cluster.means)

assert len(means) == 3

# Check that the means are close to what we expect
assert squared_distance(means[0], [-44, 5]) < 1
assert squared_distance(means[1], [-16, -10]) < 1
assert squared_distance(means[2], [18, 20]) < 1

# You find three clusters centered at [–44, 5], [–16, –10], and [18, 20], and 
# you look for meetup venues near those locations

plt.scatter(xs, ys)
for i, mean in enumerate(means):
    plt.text(mean[0], mean[1], i + 1)
plt.xlabel("blocks east of city center")
plt.ylabel("blocks north of city center")
plt.title("User Locations -- 3 Clusters")
plt.show()

# You show your results to the VP, who informs you that now she only has enough 
# budgeted for two meetups.

random.seed(0)
cluster = KMeans(2)
cluster.train(inputs)
means = sorted(cluster.means)

plt.scatter(xs, ys)
for i, mean in enumerate(means):
    plt.text(mean[0], mean[1], i + 1)
plt.xlabel("blocks east of city center")
plt.ylabel("blocks north of city center")
plt.title("User Locations -- 2 Clusters")
plt.show()

assert len(means) == 2
assert squared_distance(means[0], [-26, -5]) < 1
assert squared_distance(means[1], [18, 20]) < 1
