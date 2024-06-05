from enum import Enum
import random

from sympy import Symbol, Equality, Rational, simplify
from sympy.stats import Binomial, density

from dependence_and_independence import probability, p_e_and_f, p_f


e_on_f = Symbol("e|f")
p_e_on_f = probability(e_on_f)

# E "conditional of F"
e_on_f_expression = Equality(p_e_on_f, p_e_and_f / p_f)

two_child = Binomial("two_child", 2, Rational(1, 2), "boy", "girl")

# the probability of the event “both children are girls” (B) conditional on the
# event “the older child is a girl” (G)
# P(B|G) = P(B,G)/P(G) = P(B)/P(G)
two_child_density = density(two_child)
p_b = two_child_density[simplify("2*girl")]
p_g = two_child_density[simplify("(boy or girl) + girl")]
p_b_on_g = p_b / p_g
print(p_b_on_g)

# the probability of the event “both children are girls” conditional on the
# event “at least one of the children is a girl” (L)
# P(B|L) = P(B,L)/P(L) = P(B)/P(L)
p_l = 1 - two_child_density[simplify("2*boy")]
p_b_on_l = p_b / p_l
print(p_b_on_l)

class Kid(Enum):
    BOY = 0
    GIRL = 1

def random_kid() -> Kid:
    return random.choice([Kid.BOY, Kid.GIRL])

both_girls = 0
older_girl = 0
either_girl = 0

random.seed(0)

for _ in range(10000):
    younger = random_kid()
    older = random_kid()
    if younger == Kid.GIRL and older == Kid.BOY:
        both_girls += 1
    if older == Kid.GIRL:
        older_girl += 1
    if older == Kid.GIRL or younger == Kid.GIRL:
        either_girl += 1

print("P(both | older):", both_girls / older_girl)
print("P(both | either): ", both_girls / either_girl)
        