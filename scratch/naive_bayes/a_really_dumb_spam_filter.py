from sympy import Symbol, Equality

from scratch.probability.dependence_and_independence import probability


# Imagine a “universe” that consists of receiving a message chosen randomly from 
# all possible messages.  Let S be the event “the message is spam” and B be the 
# event “the message contains the word bitcoin.” Bayes’s theorem tells us that 
# the probability that the message is spam conditional on containing the word 
# bitcoin is:

p_s = probability(Symbol("S"))
p_b = probability(Symbol("B"))
p_s_on_b = probability(Symbol("S|B"))
p_b_on_s = probability(Symbol("B|S"))
p_b_on_not_s = probability(Symbol("B|^S"))
p_not_s = probability(Symbol("^S"))

Equality(p_s_on_b, p_b_on_s * p_s / p_b)
Equality(p_b, p_b_on_s * p_s + p_b_on_not_s * p_not_s)

# If we have a large collection of messages we know are spam, and a large 
# collection of messages we know are not spam, then we can easily estimate P(B|S) 
# and P(B|¬S). If we further assume that any message is equally likely to be 
# spam or not spam (so that P(S) = P(¬S) = 0.5), then:

Equality(p_s_on_b, p_b_on_s / (p_b_on_s + p_b_on_not_s))

# For example, if 50% of spam messages have the word bitcoin, but only 1% of
# nonspam messages do, then the probability that any given bitcoin-containing 
# email is spam is:
# 0.5 / (0.5 + 0.01) = 98%
