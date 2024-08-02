from scratch.linear_algebra.vectors import Vector, dot


# An artificial neural network (or neural network for short) is a predictive 
# model motivated by the way the brain operates.  Think of the brain as a 
# collection of neurons wired together.  Each neuron looks at the outputs of 
# the other neurons that feed into it, does a calculation, and then either fires 
# (if the calculation exceeds some threshold) or doesn’t (if it doesn’t).

# Pretty much the simplest neural network is the perceptron, which approximates 
# a single neuron with n binary inputs.  It computes a weighted sum of its 
# inputs and “fires” if that weighted sum is 0 or greater:

def step_function(x: float) -> float:
    return 1.0 if x >= 0 else 0.0

def perceptron_output(weight: Vector, bias: float, x: Vector) -> float:
    """
    Returns 1 if the perceptron 'fires', 0 if not.
    """

    calculation = dot(weight, x) + bias
    return step_function(calculation)

# The perceptron is simply distinguishing between the half-spaces separated by 
# the hyperplane of points x for which:
# dot(widght, x) + bias == 0

# With properly chosen weights, perceptrons can solve a number of simple problems. 
# For example, we can create an AND gate (which returns 1 if both its inputs 
# are 1 but returns 0 if one of its inputs is 0) with:

and_weights = [2, 2]
and_bias = -3

assert perceptron_output(and_weights, and_bias, [1, 1]) == 1
assert perceptron_output(and_weights, and_bias, [0, 1]) == 0
assert perceptron_output(and_weights, and_bias, [1, 0]) == 0
assert perceptron_output(and_weights, and_bias, [0, 0]) == 0

# If both inputs are 1, the calculation equals 2 + 2 – 3 = 1, and the output 
# is 1.  If only one of the inputs is 1, the calculation equals 2 + 0 – 3 = –1, 
# and the output is 0.  And if both of the inputs are 0, the calculation equals 
# –3, and the output is 0.

# Using similar reasoning, we could build an OR gate with:

not_weights = [-2.]
not_bias = 1.

assert perceptron_output(not_weights, not_bias, [0]) == 1
assert perceptron_output(not_weights, not_bias, [1]) == 0

# However, there are some problems that simply can’t be solved by a single 
# perceptron.  For example, no matter how hard you try, you cannot use a 
# perceptron to build an XOR gate that outputs 1 if exactly one of its inputs 
# is 1 and 0 otherwise.  This is where we start needing more complicated neural 
# networks.

# Like real neurons, artificial neurons start getting more interesting when you 
# start connecting them together.