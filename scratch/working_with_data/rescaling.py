from typing import NamedTuple, List, Tuple


from scratch.linear_algebra.vectors import distance, Vector, vector_mean
from scratch.statistics.correlation import standard_deviation


class HeighAndWeight(NamedTuple):
    height_inches: float
    height_centimeters: float
    weight_pounds: float

a = HeighAndWeight(63, 160, 150)
b = HeighAndWeight(67, 170.2, 160)
c = HeighAndWeight(70, 177.8, 171)

# if we measure height in inches, then B's nearest neighbor is A:
a_to_b = distance([a.height_inches, a.weight_pounds], [b.height_inches, b.weight_pounds])
a_to_c = distance([a.height_inches, a.weight_pounds], [c.height_inches, c.weight_pounds])
b_to_c = distance([b.height_inches, b.weight_pounds], [c.height_inches, c.weight_pounds])

print(f"a_to_b: {a_to_b}, a_to_c: {a_to_c}, b_to_c: {b_to_c}")

# however, if we measure height in centimeters, then B's nearest neighbor is instead C:
a_to_b = distance([a.height_centimeters, a.weight_pounds], [b.height_centimeters, b.weight_pounds])
a_to_c = distance([a.height_centimeters, a.weight_pounds], [c.height_centimeters, c.weight_pounds])
b_to_c = distance([b.height_centimeters, b.weight_pounds], [c.height_centimeters, c.weight_pounds])

print(f"a_to_b: {a_to_b}, a_to_c: {a_to_c}, b_to_c: {b_to_c}")

# We will sometimes rescale our data so that each dimension has mean 0 and standard 
# deviation 1.

def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
    """
    Returns the mean and standard deviation from each position.
    """

    dim = len(data[0])
    means = vector_mean(data)
    stdevs = [standard_deviation([vector[i] for vector in data])
              for i in range(dim)]

    return means, stdevs

def rescale(data: List[Vector]) -> List[Vector]:
    """
    Rescales the input data so that each position has mean 0 and standard deviation 
    1. (Leaves a position as is if its standard deviation is 0.)
    """
    
    dim = len(data[0])
    means, stdevs = scale(data)

    # Make a copy of each vector
    rescaled = [v[:] for v in data]

    for v in rescaled:
        for i in range(dim):
            if stdevs[i] > 0:
                v[i] = (v[i] - means[i]) / stdevs[i]

    return rescaled
