from typing import NamedTuple, Union, Any, List
from collections import Counter

from scratch.decision_trees.creating_a_decision_tree import inputs, Candidate, \
    partition_entropy_by, partition_by


# Now that we’ve seen how the algorithm works, we would like to implement it 
# more generally. This means we need to decide how we want to represent trees. 
# We’ll use pretty much the most lightweight representation possible. We define 
# a tree to be either:
#   * a Leaf (that predicts a single value), or
#   * a Split (containing an attribute to split on, subtrees for specific values 
#     of that attribute, and possibly a default value to use if we see an 
#     unknown value).


class Leaf(NamedTuple):
    value: Any


class Split(NamedTuple):
    attribute: str
    subtree: dict
    default_value: Any = None


DecisionTree = Union[Leaf, Split]

# There’s still the question of what to do if we encounter an unexpected (or 
# missing) attribute value.  What should our hiring tree do if it encounters a 
# candidate whose level is Intern?  We’ll handle this case by populating the 
# default_value attribute with the most common label.

# Given such a representation, we can classify an input with:

def classify(tree: DecisionTree, input: Any) -> Any:
    """
    Classify the input using the given decision tree.
    """

    # If this is a leaf node, return its value
    if isinstance(tree, Leaf):
        return tree.value

    # Otherwise this tree consists of an attribute to split on and a dictionary 
    # whose keys are values of that attribute and whose values are subtrees to 
    # consider next
    subtree_key = getattr(input, tree.attribute)
    
    if subtree_key not in tree.subtree.keys():
        return tree.default_value
    
    subtee = tree.subtree[subtree_key]
    return classify(subtee, input)

# All that's left is to build the tree representation from our training data:

def build_tree_id3(inputs: List[Any],
                   split_attributes: List[str],
                   target_attribute: str) -> DecisionTree:
    # Count target labels
    label_counts = Counter(getattr(input, target_attribute) for input in inputs)
    most_common_label = label_counts.most_common(1)[0][0]

    # If no split attributes left, return the majority label
    if not split_attributes:
        return Leaf(most_common_label)

    # Otherwise split by the best attribute
    def split_entropy(attribute: str) -> float:
        """
        Helper function for finding the best attribute.
        """

        return partition_entropy_by(inputs, attribute, target_attribute)

    best_attribute = min(split_attributes, key=split_entropy)
    partitions = partition_by(inputs, best_attribute)
    new_attributes = [a for a in split_attributes if a != best_attribute]
    
    # Recursively build the subtrees
    subtree = {k: build_tree_id3(v, new_attributes, target_attribute)
                for k, v in partitions.items()}
    return Split(best_attribute, subtree, most_common_label)

# In the tree we built, every leaf consisted entirely of True inputs or entirely 
# of False inputs.  This means that the tree predicts perfectly on the training 
# dataset.  But we can also apply it to new data that wasn’t in the training set:

tree = build_tree_id3(inputs,
                      ["level", "lang", "tweets", "phd"],
                      "did_well")

# Should predict True
assert classify(tree, Candidate("Junior", "Java", True, False))
# Should predict False
assert not classify(tree, Candidate("Junior", "Java", True, True))

# And also to data with unexpected values:
# Should predict True
assert classify(tree, Candidate("Intern", "Java", True, True))
