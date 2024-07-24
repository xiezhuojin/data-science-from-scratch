# A decision tree uses a tree structure to represent a number of possible 
# decision paths and an outcome for each path.

# At the same time, finding an “optimal” decision tree for a set of training 
# data is computationally a very hard problem.  (We will get around this by 
# trying to build a good-enough tree rather than an optimal one, although for 
# large datasets this can still be a lot of work.)  More important, it is very 
# easy (and very bad) to build decision trees that are overfitted to the 
# training data, and that don’t generalize well to unseen data.  We’ll look at 
# ways to address this.

# Most people divide decision trees into classification trees (which produce 
# categorical outputs) and regression trees (which produce numeric outputs). 
# In this chapter, we’ll focus on classification trees, and we’ll work through 
# the ID3 algorithm for learning a decision tree from a set of labeled data, 
# which should help us understand how decision trees actually work.  To make 
# things simple, we’ll restrict ourselves to problems with binary outputs like 
# “Should I hire this candidate?” or “Should I show this website visitor 
# advertisement A or advertisement B?” or “Will eating this food I found in the 
# office fridge make me sick?