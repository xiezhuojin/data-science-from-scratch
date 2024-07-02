# Imagine building a model to make a binary judgment. Is this email span? Should 
# we hire this candidate? Is this air traveler secretly a terrorist?

# Given a set of labeled data and such a predictive model, every data point lies 
# in one of four categories:
#   True positive
#   False positive
#   False negative
#   True negative

#                       Spam            Not spam
# Predict "spam"        TP              FP
# Predict "not spam"    FN              TN

# We can compute various statistics about model performance. For example, accuracy 
# is defined as the fraction of correct predictions:

def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total

# It's common to look at the combination of precision and recall. Precisin measures 
# how accurate our positive predictions were:

def precision(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fp)

# And recall measures what fraction of the positives our model identified:

def recall(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fn)

# Sometimes precision and recall are combined into the F1 score, which is defined 
# as:

def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)

    return 2 * p * r / (p + r)

# This is the harmonic mean of precision and recall and necessarily lies between 
# them.

# Usually the choice of a model involves a tradeoff between precision and recall. 
# A model that predicts "yes" when it's even a little bit confident will probably 
# have a high recall but a low precision; a model that predicts "yes" only when 
# it's extremely confident is likely to have a low recall and a high precision.
