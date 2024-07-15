from typing import List

from scratch.linear_algebra.vectors import Vector
from scratch.simple_linear_regression.the_model import total_sum_of_squares
from scratch.multiple_regression.fitting_the_model import error, inputs, daily_minutes_good, beta

# Against we can look at the R-squared:
def multiple_r_squared(xs: List[Vector], ys: List[float], beta: Vector) -> float:
    sum_of_squared_errors = sum(error(x, y, beta) ** 2 for x, y in zip(xs, ys))
    return 1.0 - sum_of_squared_errors / total_sum_of_squares(ys)

assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta) < 0.68

# Keep in mind, however, that adding new variables to a regression will 
# necessarily increase the R-squared.  After all, the simple regression model is 
# just the special case of the multiple regression model where the coefficients 
# on “work hours” and “PhD” both equal 0. The optimal multiple regression model 
# will necessarily have an error at least as small as that one.
# Because of this, in a multiple regression, we also need to look at the standard 
# errors of the coefficients, which measure how certain we are about our estimates 
# of each βi. The regression as a whole may fit our data very well, but if some 
# of the independent variables are correlated (or irrelevant), their coefficients 
# might not mean much.

# The typical approach to measuring these errors starts with another assumption—that 
# the errors εi are independent normal random variables with mean 0 and some 
# shared (unknown) standard deviation σ. In that case, we (or, more likely, our 
# statistical software) can use some linear algebra to find the standard error 
# of each coefficient.  The larger it is, the less sure our model is about that 
# coefficient.  Unfortunately, we’re not set up to do that kind of linear algebra 
# from scratch.
