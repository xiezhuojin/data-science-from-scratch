# There are a couple of further assumptions that are required for this model (and 
# our solution) to make sense.

# The first is that the columns of x are linearly independent—that there’s no 
# way to write any one as a weighted sum of some of the others.  If this 
# assumption fails, it’s impossible to estimate beta.  To see this in an extreme 
# case, imagine we had an extra field num_acquaintances in our data that for 
# every user was exactly equal to num_friends.

# Then, starting with any beta, if we add any amount to the num_friends 
# coefficient and subtract that same amount from the num_acquaintances coefficient, 
# the model’s predictions will remain unchanged. This means that there’s no way 
# to find the coefficient for num_friends.  (Usually violations of this assumption 
# won’t be so obvious.)

# The second important assumption is that the columns of x are all uncorrelated 
# with the errors ε. If this fails to be the case, our estimates of beta will be 
# systematically wrong.
