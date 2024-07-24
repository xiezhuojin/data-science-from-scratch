from typing import List, Any
import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


# In order to build a decision tree, we will need to decide what questions to 
# ask and in what order.   At each stage of the tree there are some possibilities 
# we’ve eliminated and some that we haven’t.  After learning that an animal 
# doesn’t have more than five legs, we’ve eliminated the possibility that it’s 
# a grasshopper.  We haven’t eliminated the possibility that it’s a duck. Each 
# possible question partitions the remaining possibilities according to its answer.

# Ideally, we’d like to choose questions whose answers give a lot of information 
# about what our tree should predict.  If there’s a single yes/no question for 
# which “yes” answers always correspond to True outputs and “no” answers to False 
# outputs (or vice versa), this would be an awesome question to pick. Conversely, 
# a yes/no question for which neither answer gives you much new information about 
# what the prediction should be is probably not a good choice.

# We capture this notion of “how much information” with entropy. You have probably 
# heard this term used to mean disorder. We use it to represent the uncertainty 
# associated with data.

# Imagine that we have a set S of data, each member of which is labeled as 
# belonging to one of a finite number of classes C1, ..., Cn. If all the data 
# points belong to a single class, then there is no real uncertainty, which 
# means we’d like there to be low entropy.  If the data points are evenly spread 
# across the classes, there is a lot of uncertainty and we’d like there to be 
# high entropy.

# In math terms, if pi is the proportion of data labeled as class ci, we define 
# the entropy as:
# H(S) = -p1 * log2p1 - ... -pn * log2pn
# with the (standard) convention that 0 * log0 = 0.

# Without worrying too much about the grisly details, each term -pi * log2pi 
# is non-negative and is close to 0 precisely when Pi is either close to 0 or 
# close to 1.

def entropy(class_probabilities: List[float]) -> float:
    """
    Given a list of class probabilities, compute the entropy.
    """

    return sum([-p * math.log2(p) for p in class_probabilities if p > 0])

xs = np.linspace(0, 1, 100)
ys = [entropy([x]) for x in xs]
plt.plot(xs, ys)
plt.title("A graph of -p * log2p")
plt.show()

# This means the entropy will be small when every pi is close to 0 or 1 (i.e., 
# when most of the data is in a single class), and it will be larger when many 
# of the pi’s are not close to 0 (i.e., when the data is spread across multiple 
# classes).  This is exactly the behavior we desire.

assert entropy([1]) == 0
assert entropy([0.5, 0.5]) == 1
assert 0.81 < entropy([0.25, 0.75]) < 0.82

# Our data will consist of pairs (input, label), which means that we’ll need to 
# compute the class probabilities ourselves. Notice that we don’t actually care 
# which label is associated with each probability, only what the probabilities are:

def class_probabilities(labels: List[Any]) -> List[float]:
    total_count = len(labels)
    return [count / total_count
            for count in Counter(labels).values()]

def data_entropy(labels: List[Any]) -> float:
    return entropy(class_probabilities(labels))

assert data_entropy(["a"]) == 0
assert data_entropy([True, False]) == 1
assert data_entropy([3, 4, 4, 4]) == entropy([0.25, 0.75])
