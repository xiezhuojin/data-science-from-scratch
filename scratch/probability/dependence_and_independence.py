from sympy import symbols, Function, Equality, Unequality


ef, e, f = symbols(("e\,f", "e", "f"))
probability = Function("P")
p_e_and_f = probability(ef)
p_e = probability(e)
p_f = probability(f)

# dependence
dependence = Unequality(p_e_and_f, p_e * p_f)
print(dependence)

# independence
independence = Equality(p_e_and_f, p_e * p_f)
