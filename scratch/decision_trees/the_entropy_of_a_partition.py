from typing import List, Any

from scratch.decision_trees.entropy import data_entropy


# What we’ve done so far is compute the entropy (think “uncertainty”) of a 
# single set of labeled data.  Now, each stage of a decision tree involves asking 
# a question whose answer partitions data into one or (hopefully) more subsets.

# Correspondingly, we’d like some notion of the entropy that results from 
# partitioning a set of data in a certain way.  We want a partition to have low 
# entropy if it splits the data into subsets that themselves have low entropy 
# (i.e., are highly certain), and high entropy if it contains subsets that (are 
# large and) have high entropy (i.e., are highly uncertain).

# Mathematically, if we partition our data S into subsets S1, ... , Sm containing 
# proportions q1, ..., qm of the data, then we compute the entropy of the 
# partition as a weighted sum:
# H = q1 * H(S1) + ... + qm * H(Sm)

def partition_entropy(subsets: List[List[Any]]) -> float:
    """
    Returns the entropy from this partition of data into subsets.
    """

    total_count = sum([len(subset) for subset in subsets])
    return sum([data_entropy(subset) * len(subset) / total_count
                for subset in subsets])
