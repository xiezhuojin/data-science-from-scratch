from typing import NamedTuple, Optional, TypeVar, List, Dict, Any
from collections import defaultdict

from scratch.decision_trees.the_entropy_of_a_partition import partition_entropy


class Candidate(NamedTuple):
    level: str
    lang: str
    tweets: bool
    phd: bool
    did_well: Optional[bool] = None # allow unlabeled data

                  #  level     lang     tweets  phd  did_well
inputs = [Candidate('Senior', 'Java',   False, False, False),
          Candidate('Senior', 'Java',   False, True,  False),
          Candidate('Mid',    'Python', False, False, True),
          Candidate('Junior', 'Python', False, False, True),
          Candidate('Junior', 'R',      True,  False, True),
          Candidate('Junior', 'R',      True,  True,  False),
          Candidate('Mid',    'R',      True,  True,  True),
          Candidate('Senior', 'Python', False, False, False),
          Candidate('Senior', 'R',      True,  False, True),
          Candidate('Junior', 'Python', True,  False, True),
          Candidate('Senior', 'Python', True,  True,  True),
          Candidate('Mid',    'Python', False, True,  True),
          Candidate('Mid',    'Java',   True,  False, True),
          Candidate('Junior', 'Python', False, True,  False)
         ]

# Our tree will consist of decision nodes (which ask a question and direct us 
# differently depending on the answer) and leaf nodes (which give us a prediction). 
# We will build it using the relatively simple ID3 algorithm, which operates in 
# the following manner.  Let’s say we’re given some labeled data, and a list of 
# attributes to consider branching on:

#   * If the data all have the same label, create a leaf node that predicts that 
#     label and then stop.
#   * If the list of attributes is empty (i.e., there are no more possible 
#     questions to ask), create a leaf node that predicts the most common label 
#     and then stop.
#   * Otherwise, try partitioning the data by each of the attributes.
#   * Choose the partition with the lowest partition entropy.
#   * Add a decision node based on the chosen attribute.
#   * Recur on each partitioned subset using the remaining attributes.

# This is what’s known as a “greedy” algorithm because, at each step, it chooses 
# the most immediately best option.  Given a dataset, there may be a better tree 
# with a worse-looking first move.  If so, this algorithm won’t find it. 
# Nonetheless, it is relatively easy to understand and implement, which makes it 
# a good place to begin exploring decision trees.

# Let’s manually go through these steps on the interviewee dataset. The dataset 
# has both True and False labels, and we have four attributes we can split on. 
# So our first step will be to find the partition with the least entropy. We’ll 
# start by writing a function that does the partitioning:

T = TypeVar("T") # generic type for inputs

def partition_by(inputs: List[T], attribute: str) -> Dict[Any, List[T]]:
    """
    Partition the inputs into lists based on the specified attribute.
    """

    partitions: Dict[Any, List[T]] = defaultdict(list)
    for input in inputs:
        key = getattr(input, attribute)
        partitions[key].append(input)

    return partitions

# and one that uses it to compute entropy:

def partition_entropy_by(inputs: List[Any], attribute: str, label_attribute: str) -> float:
    """
    Compute the entropy corresponding to the given partition.
    """
    
    # partitions consist of out inputs
    partitions = partition_by(inputs, attribute)

    # but partition_entropy needs just the class labels
    labels = [[getattr(input, label_attribute) for input in partition]
              for partition in partitions.values()]

    return partition_entropy(labels)

# Then we just need to find the minimum-entropy partitions for the whole dataset:

for key in ["level", "lang", "tweets", "phd"]:
    print(key, partition_entropy_by(inputs, key, "did_well"))

assert 0.69 < partition_entropy_by(inputs, "level", "did_well") < 0.70
assert 0.86 < partition_entropy_by(inputs, "lang", "did_well") < 0.87
assert 0.78 < partition_entropy_by(inputs, "tweets", "did_well") < 0.79
assert 0.89 < partition_entropy_by(inputs, "phd", "did_well") < 0.90

# The lowest entropy comes from splitting on level, so we’ll need to make a 
# subtree for each possible level value.  Every Mid candidate is labeled True, 
# which means that the Mid subtree is simply a leaf node predicting True. For 
# Senior candidates, we have a mix of Trues and Falses, so we need to split again:

senior_inputs = [input for input in inputs if input.level == "Senior"]
assert 0.4 == partition_entropy_by(senior_inputs, "lang", "did_well")
assert 0.0 == partition_entropy_by(senior_inputs, "tweets", "did_well")
assert 0.95 < partition_entropy_by(senior_inputs, "phd", "did_well") < 0.96

# This shows us that our next split should be on tweets, which results in a 
# zero-entropy partition.  For these Senior-level candidates, “yes” tweets always 
# result in True while “no” tweets always result in False.

# Finally, if we do the same thing for the Junior candidates, we end up splitting 
# on phd, after which we find that no PhD always results in True and PhD always 
# results in False.