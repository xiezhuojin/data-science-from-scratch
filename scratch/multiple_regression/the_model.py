from scratch.linear_algebra.vectors import Vector, dot


# In multiple regression the vector of parameters is usually called beta. We'll 
# want this to include the constant term as well, which we can achieve by adding 
# a column of 1s to our data:
# beta = [alpha, beta_1, ..., beta_k]
# and:
# x_i = [1, x_i1, ..., x_ik]

def predict(x: Vector, beta: Vector) -> float:
    """
    assumesd that the first element of x is 1.
    """

    return dot(x, beta)
