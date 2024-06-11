# Although the rationale for moving against the gradient is clear, how far to
# move is not. Indeed, choosing the right step size is more of an art than a
# science. Popular options include:

# * Using a fixed step size
# * Gradually shrinking the step size over time
# * At each step, choosing the step size that minimizes the value of the objective function”

# The last approach sounds great but is, in practice, a costly computation. To
# keep things simple, we’ll mostly just use a fixed step size.