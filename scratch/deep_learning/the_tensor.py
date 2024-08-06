import operator
from typing import List, Callable


# Deep learning originally referred to the application of “deep” neural networks
# (that is, networks with more than one hidden layer), although in practice the
# term now encompasses a wide variety of neural architectures.

# Previously, we made a distinction between vectors (one-dimensional arrays)
# and matrices (two-dimensional arrays). When we start working with more complicated
# neural networks, we’ll need to use higher-dimensional arrays as well.

# In many neural network libraries, n-dimensional arrays are referred to as 
# tensors, which is what we’ll call them too.

# So, like I said, we’ll just cheat:

Tensor = list

# And we’ll write a helper function to find a tensor’s shape:

def shape(tensor: Tensor) -> List[int]:
    sizes: List[int] = []
    while isinstance(tensor, list):
        sizes.append(len(tensor))
        tensor = tensor[0]
    return sizes

assert shape([1, 2, 3]) == [3]
assert shape([[1, 2], [3, 4], [5, 6]]) == [3, 2]

# Because tensors can have any number of dimensions, we’ll typically need to
# work with them recursively. We’ll do one thing in the one-dimensional case
# and recurse in the higher-dimensional case:

def is_1d(tensor: Tensor) -> bool:
    """
    If tensor[0] is a list, it's a higher-order tensor.
    """

    return not isinstance(tensor[0], list)

assert is_1d([1, 2, 3])
assert not is_1d([[1, 2], [3, 4]])

# which we can use to write a recursive tensor_sum function:

def tensor_sum(tensor: Tensor) -> float:
    """
    Sums up all the values in the tensor.
    """

    if is_1d(tensor):
        return sum(tensor)
    else:
        return sum([tensor_sum(t) for t in tensor])

assert tensor_sum([1, 2, 3]) == 6
assert tensor_sum([[1, 2], [3, 4]]) == 10

def tensor_apply(f: Callable[[float], float], tensor: Tensor) -> Tensor:
    """
    Applies f elementwise.
    """

    if is_1d(tensor):
        return [f(t) for t in tensor]
    else:
        return [tensor_apply(f, t) for t in tensor]

assert tensor_apply(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]
assert tensor_apply(lambda x: 2 * x, [[1, 2], [3, 4]]) == [[2, 4], [6, 8]]

def zeros_like(tensor: Tensor) -> Tensor:
    return tensor_apply(lambda _: .0, tensor)

assert zeros_like([1, 2, 3]) == [0, 0, 0]
assert zeros_like([[1, 2], [3, 4]]) == [[0, 0], [0, 0]]

# We’ll also need to apply a function to corresponding elements from two tensors
# (which had better be the exact same shape, although we won’t check that):

def tensor_combine(f: Callable[[float, float], float],
                   t1: Tensor,
                   t2: Tensor) -> Tensor:
    """
    Apply f to corresponding elements of t1 and t2.
    """

    if is_1d(t1):
        return [f(t1_, t2_) for t1_, t2_ in zip(t1, t2)]
    else:
        return [tensor_combine(f, t1_, t2_) for t1_, t2_ in zip(t1, t2)]

assert tensor_combine(operator.add, [1, 2, 3], [4, 5, 6]) == [5, 7, 9]
assert tensor_combine(operator.mul, [1, 2, 3], [4, 5, 6]) == [4, 10, 18]
