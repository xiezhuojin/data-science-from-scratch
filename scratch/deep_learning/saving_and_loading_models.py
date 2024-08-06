import json

from scratch.deep_learning.the_tensor import shape
from scratch.deep_learning.the_layer_abstraction import Layer

# Our deep model gets better than 92% accuracy on the test set, which is a nice 
# improvement from the simple logistic model.

def save_weights(model: Layer, filename: str) -> None:
    weights = list(model.params())
    with open(filename, "w") as f:
        json.dump(weights, f)

# Loading the weights back is only a little more work. We just use json.load to 
# get the list of weights back from the file and slice assignment to set the 
# weights of our model.

# (In particular, this means that we have to instantiate the model ourselves and 
# then load the weights. An alternative approach would be to also save some 
# representation of the model architecture and use that to instantiate the model. 
# That’s not a terrible idea, but it would require a lot more code and changes 
# to all our Layers, so we’ll stick with the simpler way.)

# Before we load the weights, we’d like to check that they have the same shapes 
# as the model params we’re loading them into. (This is a safeguard against, for 
# example, trying to load the weights for a saved deep network into a shallow 
# network, or similar issues.)

def load_weights(model: Layer, filename: str) -> None:
    with open(filename) as f:
        weights = json.load(f)

    # Check for consistency
    assert all(shape(param) == shape(weight)
               for param, weight in zip(model.params(), weights))

    # Then load using slice assignment
    for param, weight in zip(model.params(), weights):
        param[:] = weight
