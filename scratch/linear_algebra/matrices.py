from typing import List, Tuple, Callable
from itertools import chain

from scratch.linear_algebra.vectors import Vector


Matric = List[List[float]]

A = [ # A has 2 rows and 3 columns
    [1, 2, 3],
    [4, 5, 6]
]

B = [ # B has 3 rows and 2 columns
    [1, 2],
    [3, 4],
    [5, 6]
]

def shape(A: Matric) -> Tuple[int, int]:
    """
    Return (# of rows of A, # of columns of A).
    """

    num_rows = len(A)
    num_cols = len(A[0]) if len(A) else 0 # number of elements in first row
    return num_rows, num_cols

def get_row(A: Matric, i: int) -> Vector:
    """
    Returns the i-th row of A (as a Vector).
    """

    return A[i]

def get_column(A: Matric, j: int) -> Vector:
    """
    Return the j-th column of A (as a Vector).
    """

    return [row[j] for row in A]

def make_matric(num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]) -> Matric:
    """
    Returns a num_row x num_cols matrix, whose (i, j)-the entry is entry_fn(i, j)
    """

    return [
        [entry_fn(i, j) for j in range(num_cols)]
        for i in range(num_rows)
    ]

def identity_matrix(n: int) -> Matric:
    return make_matric(n, n, lambda i, j: 1 if i == j else 0)

assert shape(A) == (2, 3)
assert identity_matrix(5) == [
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
]

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
               (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]
size = len(set(chain.from_iterable(friendships)))
friend_matric = make_matric(size, size,
                            lambda i, j: 1 if (i, j) in friendships or (j, i) in friendships else 0)
assert friend_matric[0][2] == 1, "0 and 2 are friends"
assert friend_matric[0][0] == 0, "0 and 8 are not frinds"

# only need to look at one row
friends_of_five = [i for i, col in enumerate(friend_matric[5]) if col == 1]
assert friends_of_five == [4, 6, 7]