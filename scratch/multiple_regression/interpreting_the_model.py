# You should think of the coefficients of the model as representing 
# all-else-being-equal estimates of the impacts of each factor.  All else being 
# equal, each additional friend corresponds to an extra minute spent on the site 
# each day.  All else being equal, each additional hour in a user’s workday
# corresponds to about two fewer minutes spent on the site each day.  All else 
# being equal, having a PhD is associated with spending an extra minute on the 
# site each day.

# What this doesn’t (directly) tell us is anything about the interactions among 
# the variables. It’s possible that the effect of work hours is different for 
# people with many friends than it is for people with few friends. This model 
# doesn’t capture that. One way to handle this case is to introduce a new variable 
# that is the product of “friends” and “work hours.” This effectively allows the 
# “work hours” coefficient to increase (or decrease) as the number of friends 
# increases.

# Or it’s possible that the more friends you have, the more time you spend on the 
# site up to a point, after which further friends cause you to spend less time 
# on the site.  (Perhaps with too many friends the experience is just too 
# overwhelming?)  We could try to capture this in our model by adding another variable
# that’s the square of the number of friends.

# Once we start adding variables, we need to worry about whether their 
# coefficients “matter.”  There are no limits to the numbers of products, logs, 
# squares, and higher powers we could add.