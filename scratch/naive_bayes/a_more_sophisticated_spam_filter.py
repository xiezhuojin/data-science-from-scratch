# Imagine now that we have a vocabulary of many words, w1 ..., wn. To move this 
# into the realm of probability theory, we’ll write Xi for the event “a message 
# contains the word wi. Also imagine that (through some unspecified-at-this-point 
# process) we’ve come up with an estimate P(Xi|S) for the probability that a 
# spam message contains the ith word, and a similar estimate P(Xi|¬S) for the 
# probability that a nonspam message contains the ith word.

# The key to Naive Bayes is making the (big) assumption that the presences (or 
# absences) of each word are independent of one another, conditional on a message 
# being spam or not. Intuitively, this assumption means that knowing whether a 
# certain spam message contains the word bitcoin gives you no information about 
# whether that same message contains the word rolex.

# This is an extreme assumption.  (There’s a reason the technique has naive in 
# its name.) Imagine that our vocabulary consists only of the words bitcoin and 
# rolex, and that half of all spam messages are for “earn bitcoin” and that the 
# other half are for “authentic rolex.”  In this case, the Naive Bayes estimate 
# that a spam message contains both bitcoin and rolex is:
# P(X1 = 'bitcoin', X2 = 'rolex' | S) = P(X1 = 'bitcon' | S) * P(X2 = 'rolex' | S) = 0.5 * 0.5 = 0.25

# since we’ve assumed away the knowledge that bitcoin and rolex actually never 
# occur together.  Despite the unrealisticness of this assumption, this model 
# often performs well and has historically been used in actual spam filters.

# The same Bayes’s theorem reasoning we used for our “bitcoin-only” spam filter 
# tells us that we can calculate the probability a message is spam using the 
# equation: P(S|X = x) = P(X = x|S) / [P(X = x|S) + P(X = x|^S)]

# The Naive Bayes assumption allows us to compute each of the probabilities on 
# the right simply by multiplying together the individual probability estimates 
# for each vocabulary word.

# The only challenge left is coming up with estimates for P(Xi|S) and P(Xi|^S), 
# the probabilities that a spam message (or nonspam message) contains the word 
# wi. If we have a fair number of “training” messages labeled as spam and not 
# spam, an obvious first try is to estimate P(Xi|S) simply as the fraction of 
# spam messages containing the word wi.

# This causes a big problem, though.  Imagine that in our training set the 
# vocabulary word data only occurs in nonspam messages. Then we’d estimate 
# P(data|S) = 0. The result is that our Naive Bayes classifier would always
# assign spam probability 0 to any message containing the word data, even a 
# message like “data on free bitcoin and authentic rolex watches. To avoid this 
# problem, we usually use some kind of smoothing.

# In particular, we’ll choose a pseudocount—k—and estimate the probability of
# seeing the ith word in a spam message as:
# P(Xi|S) = (k + numberof spamas containing wi) / (2k + number of spams)

# We do similarly for P(Xi|^S). That is, when computing the spam probabilities 
# for the ith word, we assume we also saw k additional nonspams containing the 
# word and k additional nonspams not containing the word.