import math
from typing import NamedTuple

import numpy as np
from sympy import Symbol, Eq, Matrix, solve
import matplotlib.pyplot as plt
from sklearn import svm

from scratch.linear_algebra.vectors import Vector, dot
from scratch.logistic_regression.the_problem import xs, \
    paid_years_experience, paid_annual_salary, \
    unpaid_annual_salary, unpaid_years_experience
from scratch.logistic_regression.applying_the_model import beta_unscaled

# The set of points where dot(beta, x_i) equals 0 is the boundary between our 
# classes.  We can plot this to see exactly what our model is doing.
def get_annual_salary(beta: Vector, year_experience: float) -> float:
    y = 0
    y = (0 - dot(beta, [1, year_experience, y])) / beta[2]
    return y

max_year_experience = math.floor(max([x[1] for x in xs]) + 1)
hyperplance_xs = np.linspace(0, max_year_experience, 100)
hyperplane_ys = [get_annual_salary(beta_unscaled, x) for x in hyperplance_xs]

plt.scatter(paid_years_experience, paid_annual_salary, marker="o")
plt.scatter(unpaid_years_experience, unpaid_annual_salary, marker="+")
plt.plot(hyperplance_xs, hyperplane_ys)
plt.xlabel("years experience")
plt.ylabel("annual salary")
plt.title("Paid and unpiad users with decision boundary")
plt.legend()
plt.show()

# This boundary is a hyperplane that splits the parameter space into two 
# half-spaces corresponding to predict paid and predict unpaid. We found it as 
# a side effect of finding the most likely logistic model.

# An alternative approach to classification is to just look for the hyperplane 
# that “best” separates the classes in the training data. This is the idea 
# behind the support vector machine, which finds the hyperplane that maximizes 
# the distance to the nearest point in each class.

# Finding such a hyperplane is an optimization problem that involves techniques 
# that are too advanced for us. A different problem is that a separating 
# hyperplane might not exist at all. In our “who pays?” dataset there simply 
# is no line that perfectly separates the paid users from the unpaid users.

class ExampleData(NamedTuple):
    value: int
    positive: int


dataset = [
    ExampleData(-4, 1),
    ExampleData(-3, 1),
    ExampleData(-2, 0),
    ExampleData(-1, 0),
    ExampleData(0, 0),
    ExampleData(1, 0),
    ExampleData(2, 0),
    ExampleData(3, 1),
    ExampleData(3, 1),
    ExampleData(4, 1),
]

positives = [d.value for d in dataset if d.positive]
nagatives = [d.value for d in dataset if not d.positive]
plt.scatter(positives, [0 for _ in positives], marker="+")
plt.scatter(nagatives, [0 for _ in nagatives], marker="o")
plt.title("A nonseparable one-dimensional dataset")
plt.show()

# We can sometimes get around this by transforming the data into a higher-dimensional 
# space.

mapped_dataset = np.array([(d.value, d.value ** 2, d.positive) for d in dataset])
x_train = mapped_dataset[:, :2]
y_train = mapped_dataset[:, -1]

clf = svm.SVC(kernel="linear")
clf.fit(x_train, y_train)

w = Matrix(clf.coef_[0])
b = clf.intercept_[0]
x1 = Symbol("x1")
x2 = Symbol("x2")
x = Matrix((x1, x2))
equation = Eq(w.dot(x) + b, 0)

x1_ = np.linspace(-5, 5, 100)
x2_ = [solve(equation.subs(x1, x), x2) for x in x1_]
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.scatter(x1_, x2_)
plt.xlim([-6, 6])
plt.ylim([-5, 20])
plt.show()

# It’s clear that there’s no hyperplane that separates the positive examples 
# from the negative ones.  However, look at what happens when we map this dataset 
# to two dimensions by sending the point x to (x, x**2). Suddenly it’s possible 
# to find a hyperplane that splits the data.

# This is usually called the kernel trick because rather than actually mapping 
# the points into the higher-dimensional space (which could be expensive if there 
# are a lot of points and the mapping is complicated), we can use a “kernel” 
# function to compute dot products in the higher-dimensional space and use those 
# to find a hyperplane.
