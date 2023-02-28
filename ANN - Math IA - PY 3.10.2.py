import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt # convention on importing pyplot
from tqdm import tqdm

# We assume that cats are 2D data points centered around the point of coordinate (2,2) and are normally distributed and dogs are also normally distributed but around the points (-2,-2)
# The reason for both cats and dogs being 2D data points is that it corresponds to 2 features describing these animals, such features could be for example Eyes Spacing, Weights, Heights, Ears, etc...
# Let us consider the previous described data model (points)
# Simulating/Generating TRAINING data (Used to find the value of the weights (Cats)

n_train = 1000

# Simulating/Generating TRAINING data (Used to find the value of the weights (Cats)
training_cats = np.random.randn(n_train , 2 ) + 2 * np.ones(( 1 , 1 )) # pos
# Pos stands for positive

# print(training_cats[:10]) # Print statement for debugging
# print(training_cats.shape) # Print statement for debugging
# print(np.mean(training_cats, 0)) # Print statement for debugging

# Simulating/Generating TRAINING data (Used to find the value of the weights (Dogs)
training_dogs = np.random.randn(n_train , 2 ) - 2 * np.ones(( 1 , 1 )) # neg
# neg stands for negative

training_cats = np.concatenate([training_cats , [[ 1 ]] * n_train] , 1 )
training_dogs = np.concatenate([training_dogs , [[ 0 ]] * n_train] , 1 )

# Concatenating the training data
training_data = np.concatenate([training_cats , training_dogs] , 0 )

plt.scatter(training_data[: , 0 ] , training_data[: , 1 ] , c =training_data[: , 2 ] , s = 1 )
plt.title( 'Cats & Dogs' )
plt.suptitle( 'Training Data' )
ax = plt.gca()
ax.spines[ 'right' ].set_color( 'none' )
ax.spines[ 'top' ].set_color( 'none' )
ax.xaxis.set_ticks_position( 'bottom' )
ax.spines[ 'bottom' ].set_position(( 'data' , 0 ))
ax.spines[ 'bottom' ].set_color(( 'red' ))
ax.yaxis.set_ticks_position( 'left' )
ax.spines[ 'left' ].set_position(( 'data' , 0 ))
ax.spines[ 'left' ].set_color(( 'red' ))
plt.xticks( fontsize = 15 , rotation = 45 )
plt.yticks( fontsize = 15 , rotation = 45 )
plt.show()

# (.randn is to make normal distribution)
# (n_train is the number of samples)
# 2 is the size of the sample (2 parameters / 2D data points))

n_test = 100

# Simulating/Generating TESTING data (Used to evaluate the value of the weights/model (Cats))
testing_cats = np.random.randn(n_test , 2 ) + 2 * np.ones(( 1 , 1 ))
# pos

# Simulating/Generating TESTING data (Used to evaluate thevalue of the weights/model (Cats))
testing_dogs = np.random.randn(n_test , 2 ) - 2 * np.ones(( 1 , 1 ))
# neg

testing_cats = np.concatenate([testing_cats , [[ 1 ]] * n_test] , 1 )
testing_dogs = np.concatenate([testing_dogs , [[ 0 ]] * n_test] , 1 )
testing_data = np.concatenate([testing_cats , testing_dogs] , 0 )

np.random.shuffle(testing_data)

plt.scatter(testing_data[: , 0 ] , testing_data[: , 1 ] , c =testing_data[: , 2 ] , s = 1 )
ax = plt.gca()
plt.title( 'Cats & Dogs' )
plt.suptitle( 'Testing Data' )
ax.spines[ 'right' ].set_color( 'none' )
ax.spines[ 'top' ].set_color( 'none' )
ax.xaxis.set_ticks_position( 'bottom' )
ax.spines[ 'bottom' ].set_position(( 'data' , 0 ))
ax.spines[ 'bottom' ].set_color(( 'red' ))
ax.yaxis.set_ticks_position( 'left' )
ax.spines[ 'left' ].set_position(( 'data' , 0 ))
ax.spines[ 'left' ].set_color(( 'red' ))
plt.xticks( fontsize = 15 , rotation = 45 )
plt.yticks( fontsize = 15 , rotation = 45 )
plt.show()

def sigmoid (x):
    return 1 / ( 1 + np.exp(-x))

def derivative_sigmoid (x):
    return np.exp(-x) / ( 1 + np.exp(-x)) ** 2

class Perceptron():
    def __init__ ( self , weights , bias):
        self .weights = np.asarray(weights)
        self .bias = bias
    def forward ( self , x1 , x2):
        middleP = x1 * self.weights[0] + x2 * self.weights[1] + self.bias # scalar product + bias 

        outputP = sigmoid(middleP) # Activation function
        if outputP > 0.5 : # Classification function
            outputC = 1
        else :
            outputC = 0
        # Local gradients
        lg = np.asarray([
            x1 * derivative_sigmoid(middleP) ,
            x2 * derivative_sigmoid(middleP) ,
            derivative_sigmoid(middleP)
        ])
        return outputP , outputC , lg
    def backprop ( self , derr_out , lg):
        # backpropagates the gradients
        return derr_out * lg
    def update ( self , descent):
        # update the weights by the gradient descent method
        self .weights -= descent[: 2 ]
        self .bias -= descent[- 1 ]
    def get_weights ( self ):
        return self .weights , self .bias

# ------------------- ANN DEFINITION -------------------
# ------------------------------------------------------
lr = 1 # learning rate
bias = np.random.rand() # Value bias
weights = [np.random.rand() , np.random.rand()] # Weights
oneP_ann = Perceptron(weights , bias)

# ------------------- TRAINING + EPOCH EVALUATION -------------------
# -------------------------------------------------------------------
epochs = 1 # number of epochs
batch = 1 # size of the batches
# number of batches and training data reduction
n_batches = int (training_data.shape[ 0 ] // batch)
for epoch in range (epochs):
    # shuffling, and batching the training data
    np.random.shuffle(training_data)
    etraining_data = training_data[:batch * n_batches] # reduction
    etraining_data = etraining_data.reshape(n_batches , batch , 3 ) # batching
    pbar = tqdm( total = len (etraining_data) , ascii = True, desc = "Epoch {} / Error {}" .format(epoch + 1 , np.nan))
    for training_batch in etraining_data:
        errors = []
        gradients = []
        for sample in training_batch:
            # sample[:2] is the input and sample[-1] is the objective for output
            outputP , _ , lg = oneP_ann.forward(*sample[: 2 ]) # apply the ann on the input
            error = (outputP - sample[- 1 ]) ** 2 # compute the error
            errors.append(error)
            derror = 2 * (outputP - sample[- 1 ]) # compute the derivative of the error
            gradient = oneP_ann.backprop(derror , lg) # backpropagate the derivative to the weights
            gradients.append(gradient)
        gradients = np.array(gradients)
        batch_error = np.mean(errors)
        batch_gradients = np.mean(gradients , 0 )
        # print(batch_gradients)
        # update the weights
        oneP_ann.update(batch_gradients)
        # print(oneP_ann.get_weights())
        pbar.set_description_str( desc = "Epoch {} / Error {}" .format(epoch + 1 , batch_error))
        pbar.update()
    pbar.close()
    testing_goals = []
    testing_results = []
    for sample in testing_data:
        testing_goals.append(sample[- 1 ])
        _ , outputC , _ = oneP_ann.forward(*sample[: 2 ])
        testing_results.append(outputC)
    testing_goals = np.asarray(testing_goals)
    testing_results = np.asarray(testing_results)
    confusion_matrix = pd.crosstab(
        testing_goals ,
        testing_results ,
        rownames =[ 'Actual' ] ,
        colnames =[ 'Predicted' ])
    print ( " \n " )
    print (confusion_matrix)
    print("Epoch {} Test acccuracy {}% \n\n ".format(epoch, np.sum(testing_goals == testing_results) / len(testing_goals) * 100))
