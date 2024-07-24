# Given how closely decision trees can fit themselves to their training data,
# it’s not surprising that they have a tendency to overfit.  One way of avoiding
# this is a technique called random forests, in which we build multiple decision
# trees and combine their outputs. If they’re classification trees, we might let 
# them vote; if they’re regression trees, we might average their predictions.

# Our tree-building process was deterministic, so how do we get random trees?

# One piece involves bootstrapping data (recall “Digression: The Bootstrap”). 
# Rather than training each tree on all the inputs in the training set, we train 
# each tree on the result of bootstrap_sample(inputs). Since each tree is built 
# using different data, each tree will be different from every other tree. 
# (A side benefit is that it’s totally fair to use the nonsampled data to test 
# each tree, which means you can get away with using all of your data as the 
#training set if you are clever in how you measure performance.) This technique 
# is known as bootstrap aggregating or bagging.

    # # if there are already few enough split candidates, look at all of them
    # if len(split_candidates) <= self.num_split_candidates:
    #     sampled_split_candidates = split_candidates
    # # otherwise pick a random sample
    # else:
    #     sampled_split_candidates = random.sample(split_candidates,
    #                                              self.num_split_candidates)

    # # now choose the best attribute only from those candidates
    # best_attribute = min(sampled_split_candidates, key=split_entropy)

    # partitions = partition_by(inputs, best_attribute)

# This is an example of a broader technique called ensemble learning in which 
# we combine several weak learners (typically high-bias, low-variance models)
# in order to produce an overall strong model.