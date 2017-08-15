

import numpy as np

# This code is a definition of the sigmoid function, which is the type of non-linearity chosen for this neural net. It is not the only type of non-linearity that can be chosen, but is has nice analytical features and is easy to teach with.

def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# The following code creates the input matrix (third node is given 1 for all inputs to handle case of all zeros for first input)

X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# The output of the OR function follows.

y = np.array([[0],
              [1],
              [1],
              [1]])

# The seed for the random generator to return the same random numbers each time for being deterministic, which is very useful for debugging.
np.random.seed(1)

# Initialization of weights to random numbers. syn0 is weight matrix between input layer and output layer. since our network is perception so it has one input layer and one output layout
syn0 = 2 * np.random.random((3, 1)) - 1


# This is iteration training loop for network training. error decreases on each cycle of training by the slop of sigmoid function.
iterations = 100000
for j in range(iterations):

    # Calculating forward through out the network
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    # Calculating error
    l1_error = y - l1
    if(j % 10000) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output.
        print("Error: " + str(np.mean(np.abs(l1_error))))

    # Calculating delta by multiplying error by the slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)

    # Updating weights (no alpha learning term here..), by multiplying delta with input matrix effects to only update weights for non-zero inputs
    # W = W + alpha.input.error
    syn0 += np.dot(l0.T, l1_delta)

print ("Output After Training:")
print (l1)
