# Another way of thinking about the overfitting problem is as a tradeoff between 
# bias and variance.

# Both are measures of what would happen if you were to retrain your model many 
# times on different sets of training data (from the same larger population). 

# Thinking about model problems this way can help you figure out what to do when 
# your model doesn't work so well. 

# If your model has high bias (which means it performs poorly even on your training 
# data), one thing to try is adding more features. If your model has high variance, 
# you can similarly remove features. But another solution is to obtain more data 
# (if you can).