from sympy import Symbol, Equality, Rational

from scratch.probability.dependence_and_independence import p_e, p_f, p_e_and_f, probability
from scratch.probability.conditional_probability import p_e_on_f

f_on_e = Symbol("f|e")
f_on_not_e = Symbol("f|^e")
p_f_on_e = probability(f_on_e)
p_f_on_not_e = probability(f_on_not_e)

equaltion1 = Equality(p_e_on_f, p_e_and_f / p_f)
equaltion2 = equaltion1.subs(p_e_and_f, p_f_on_e * p_e)
equaltion3 = equaltion2.subs(p_f, p_f_on_e + p_f_on_not_e)
# print(equaltion3)

# a disease that affects 1 in every 10,000 people
# a test for this disease that gives the correct result 99% of the time
# Let’s use T for the event “your test is positive” and D for the event “you have the disease.”
# Then Bayes’s theorem says that the probability that you have the disease, conditional on testing positive, is:”
# P(D|T) = P(D,T) / P(T) = P(T|D) * P(D) / P(T) = P(T|D) * P(D) / (P(T|^D) + P(T|D))
# = (99/100) * (1/10000) / ((1 - 99/100) + (1/10000))
p = Rational(99, 100) * Rational(1, 10_000) / (1 - Rational(99, 100) + Rational(1, 10_000))
# print(p)