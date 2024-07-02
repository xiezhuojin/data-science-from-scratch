# As has been mentioned, when your data doesn't have enough features, your model 
# is likely to underfit. And when your data has too many features, it's easy to 
# overfit. But what are features, and where do they come from?

# Features are whatever inputs we provide to our model.

# In the simplest case, features are simply given to you. If you want to predict 
# someone's salary based on her years of experience, then years of experience is 
# the only feature you have.

# Things become more interesting as your data becomes more complicated, Imagine 
# trying to build a spam filter to predict whether an email is junk or not. Most 
# models won't know what to do with a raw email, which is just a collection of 
# text. You'll have to extract features. For example:
#   * Does the email contain the word Viagra?
#   * How many times does the letter d appear?
#   * What was the domain of the sender?

# The answer to a question like the first question here is simply a yes or no, 
# which we typically encode as a 1 or 0. The second is a number. And the third 
# is choice from a discrete set of options.

# How do we choose features? That's where a combination fo experience and domain 
# expertise comes to play.