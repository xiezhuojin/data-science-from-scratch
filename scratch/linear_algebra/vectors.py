import math
from typing import List


Vector = List[float]

height_weight_age = [
    70, # inches
    170, # pounds
    40, # years
]

grades = [
    95, # exam1
    80, # exam2
    75, # exam3
    62, # exam4
]

def add(v: Vector, w: Vector) -> Vector:
    """
    Adds corresponding elements.
    """

    assert len(v) == len(w)
    return [v_ + w_ for v_, w_ in zip(v, w)]

def subtract(v: Vector, w: Vector) -> Vector:
    """
    Subtracts corresponding elements.
    """

    assert len(v) == len(w)
    return [v_ - w_ for v_, w_ in zip(v, w)]

def vector_sum(vectors: List[Vector]) -> Vector:
    """
    Sums all corresponding elements.
    """

    # check that vectors is not empty
    assert len(vectors) != 0
    # check the vectors are all the same size
    num_elements = [len(vector) for vector in vectors]
    for i in range(1, len(num_elements)):
        assert num_elements[i] == num_elements[0]

    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors) for i in range(len(vectors[0]))]

def scaler_multiply(c: float, v: Vector) -> Vector:
    """
    Multiplies every element by c.
    """

    return [c * v_ for v_ in v]

def vector_mean(vectors: List[Vector]) -> Vector:
    """
    Compute the element-wise average.
    """

    n = len(vectors)
    return scaler_multiply(1 / n, vector_sum(vectors))

def dot(v: Vector, w: Vector) -> float:
    """
    Computes v_1 * w_1 + ... + v_n * w_n.
    """

    assert len(v) == len(w)
    return sum([v_ * w_ for v_, w_ in zip(v, w)])

def sum_of_squares(v: Vector) -> float:
    """
    Returns v_1 * v1 + ... + v_n * v_n.
    """

    return dot(v, v)

def magnitude(v: Vector) -> float:
    """
    Returns the magnitude (or length) of v
    """

    return math.sqrt(sum_of_squares(v))

def squared_distance(v: Vector, w: Vector) -> float:
    """
    Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2.
    """

    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
    """
    Computes the distance between v and w.
    """

    return magnitude(subtract(v, w))

assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]
assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]
assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]
assert scaler_multiply(2, [1, 2, 3]) == [2, 4, 6]
assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]
assert dot([1, 2, 3], [4, 5, 6]) == 32
assert magnitude([3, 4]) == 5
