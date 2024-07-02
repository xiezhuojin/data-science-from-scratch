import random
from typing import TypeVar, List, Tuple


# A common danger in machine learning is overfitting - producing a model that 
# performs well on the data you train it on but generalizes poorly to any new Data. 

# The other side of this is underfitting - producing a model that doesn't perform 
# well even on the training data, although typically when this happens you decide 
# your model isn't goo enough and keep looking for a better one.

# Clearly, models that are too complex lead to overfitting and don't generalize 
# well beyond the data they were trained on. So how do we make sure out models 
# aren't too complex? The most fundamental approach involves using different data 
# to train the model and to test the model.

# The simplest way to do ths is split the dataset, so that (for example) two-thirds 
# of it is used to train the model, after which we measure the model's performance 
# on the remaining third:

X = TypeVar("X") # generic type to represent a data point

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """
    Split data into fractions [prob, 1 - prob].
    """

    data = data[:]
    random.shuffle(data)
    cut = int(len(data) * prob)

    return data[:cut], data[cut:]

data = [n for n in range(1000)]
train, test = split_data(data, 0.75)

# The proportions should be correct
assert len(train) == 750
assert len(test) == 250
assert sorted(train + test) == data

# Often, we'll have paired input variables and output variables. In that case, 
# we need to make sure to put corresponding values together in either the training 
# data or the test data:

Y = TypeVar("Y")

def train_test_split(xs: List[X],
                     ys: List[Y],
                     test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    # Generate the indeces and split them
    indx = [i for i in range(len(xs))]
    train_idx, test_idx = split_data(indx, 1 - test_pct)
    return (
        [xs[i] for i in train_idx], # x_train
        [xs[i] for i in test_idx], # x_test
        [ys[i] for i in train_idx], # y_train
        [ys[i] for i in test_idx] # y_test
    )

xs = [x for x in range(1000)]
ys = [2 * x for x in xs]
x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.25)

assert len(x_train) == len(y_train) == 750
assert len(x_test) == len(y_test) == 250

assert all(2 * x == y for x, y in zip(x_train, y_train))
assert all(2 * x == y for x, y in zip(x_test, y_test))

# If the model was overfit to the training data, then it will hopefully perform 
# really poorly on the (completely separate) test data. Said differently, if 
# performs well on the test data, then you can be more confident that it's fitting 
# rather than overfitting.

# However, there are a couple ouf ways this can go wrong.

# The first is if there are common patterns in the test and traning data that 
# wouldn't generalize to a larger dataset.

# For example, imagine that your dataset consists of user activity, with one row 
# per user per week. In such a case, most users will appear in both the training 
# data and the test data, and certain models might learn to identify users rather 
# that discover relationships involving attributes. This isn't a huge worry, 
# although it dis happen to me once.

# A bigger problem is if you use the test/train split not just to just a model 
# but also to choose from among many models. In that case, although each individual 
# model may not be overfit, "choosing a model that performs best on the test set" 
# is a meta-training formed best on the test set is going to perform well on the 
# test set.)

# In such a situation, you should split the data into three parts: a training set 
# for building models, a validation set for choosing among trained models, and a 
# test test for judging the final model.
