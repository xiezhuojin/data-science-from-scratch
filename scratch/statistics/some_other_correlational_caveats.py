import matplotlib.pyplot as plt

from correlation import correlation


# A correlation of zero indicates that there is no linear relationship between
# the two variables.

x = [-2, -1, 0, 1, 2]
y = [ 2,  1, 0, 1, 2]

assert correlation(x, y) == 0

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# x and y have zero correlation. But they certainly have a relationship—each
# element of y equals the absolute value of the corresponding element of x.
x = [-2, -1, 0, 1, 2]
y = [99.98, 99.99, 100, 100.01, 100.02]
assert correlation(x, y) > 0

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()