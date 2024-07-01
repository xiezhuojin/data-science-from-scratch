from typing import List, NamedTuple
from collections import Counter

from sklearn.neighbors import KNeighborsClassifier

from scratch.linear_algebra.vectors import Vector, distance


# Nearest neighbors is one of the simplest predictive models there is. It makes 
# no mathematical assumptions, and it doesn't require any sort of heavy machinery. 
# The only things it requires are:
#   * Some notion of distance
#   * An assumption that points that are close to one another are similar

def raw_majority_vote(labels: List[str]) -> str:
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner

# But this doesn't do anything intelligent with ties. For example, imagine we're 
# rating movies and the five nearest movies are rated G, G, PG, PG, and R. Then 
# G has two votes and PG also has two votes. In this case, we have several options:
#   * Pick one of the winners at random.
#   * Weight the votes by distance and pick the weighted winner.
#   * Reduce k until we find a unique winner.
# We' ll implement the third:

def majority_vote(labels: List[str]) -> str:
    """
    Assumes that labels are ordered from nearest to farthest.
    """
    
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])
    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1])

# Tie, so look at first 4, then 'b'
assert majority_vote(["a", "b", "c", "b", "a"]) == "b"

# This approach is sure to work eventually, since in the worst case we go all the 
# way down to just on label, at which point that one label wins.

# With this funciton it's easy to create a classifier:

class LabeledPoint(NamedTuple):
    point: Vector
    label: str


def knn_classify(k: int, labeled_points: List[LabeledPoint], new_point: Vector) -> str:
    # Order the labeled points from nearest to farthest.
    by_distance = sorted(labeled_points, key=lambda lp: distance(lp.point, new_point))
    # Find the labels for the k closest.
    k_nearest_labels = [lp.label for lp in by_distance[:k]]
    # and let them vote.
    return majority_vote(k_nearest_labels)

# scikit-learn can do the same
KNeighborsClassifier