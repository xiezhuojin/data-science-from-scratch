import os
from typing import Dict, List, Tuple
import csv
from collections import defaultdict
import random

import matplotlib.pyplot as plt

from scratch.k_nearest_neighbors.the_model import LabeledPoint, knn_classify
from scratch.linear_algebra.vectors import Vector
from scratch.machine_learning.overfitting_and_underfitting import split_data

# Example: The Iris Dataset

# The Iris dataset is a staple of machine learning. It contains a bunch of measurements 
# for 150 flowers representing three species of iris. For each flower we have its 
# petal length, petal width, sepal length, and sepal width, as well as its species.

# The data is comma-separated, with fiels:
# sepal_length, sepal_width, petal_length, petal_width, class

# In this section we'll try to build a model that can predict the class (that is, 
# the species) fromt he first four measurements.

def parse_iris_row(row: List[str]) -> LabeledPoint:
    """
    sepal_length, sepal_width, petal_length, petal_width, class
    """

    measurements = [float(value) for value in row[:-1]]
    label = row[-1].split("-")[-1]

    return LabeledPoint(measurements, label)

with open(os.path.join(os.path.dirname(__file__), "iris.data")) as f:
    reader = csv.reader(f)
    iris_data = [parse_iris_row(row) for row in reader if row]

# We'll also gorup just the points by species/label so we can plot them
points_by_species: Dict[str, List[Vector]] = defaultdict(list)
for iris in iris_data:
    points_by_species[iris.label].append(iris.point)

# We'd like to plot the measurements so we can see how they vary by species. 
# Unfortunately, they are four-dimensional, which makes them tricky to plot. One 
# thing we can do is look at the scatterplots for each of the six pairs of 
# measurements.

metrics = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
pairs = [(i, j) for i in range(len(metrics)) for j in range(len(metrics)) if i < j]
marks = ["+", ".", "x"] # we have 3 classes, so 3 markers

fig, ax = plt.subplots(2, 3)

for row in range(2):
    for col in range(3):
        i, j = pairs[3 * row + col]
        ax[row][col].set_title(f"{metrics[i], metrics[j]}", fontsize=8)
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])

        for mark, (species, points) in zip(marks, points_by_species.items()):
            xs = [point[i] for point in points]
            ys = [point[j] for point in points]
            ax[row][col].scatter(xs, ys, marker=mark, label=species)

ax[-1][-1].legend(loc="lower right", prop={"size": 6})
plt.show()

# If you look at those plots, it seems like the measurements really do cluster by 
# species.

# To start with, let's split the data into a test set and trainning set:
random.seed(12)
iris_train, iris_test = split_data(iris_data, 0.7)
assert len(iris_train) == 0.7 * 150
assert len(iris_test) == 0.3 * 150

# The trainning set will be the "neighbors" that we'll use to classify the points 
# in the test set. We just need to choose a value for k, the number of neighbors 
# who get to vote. Too small (think k = 1), and we let outliers have too much 
# influence; too large (think k = 105), and we just predict the most common class 
# in the dataset.

# In a real application (and with more data), we might create a separate validation 
# set and use it to choose k, Here we'll just use k = 5:

# track how many times we see (predicted, actual)
confusion_matrics: Dict[Tuple[str, str], int] = defaultdict(int)
num_correct = 0

for iris in iris_test:
    predicted = knn_classify(5, iris_train, iris.point)
    actual = iris.label

    if predicted == actual:
        num_correct += 1
    
    confusion_matrics[(predicted, actual)] += 1

pct_correct = num_correct / len(iris_test)
print(pct_correct, confusion_matrics)