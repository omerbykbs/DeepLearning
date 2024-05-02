#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: omerfaruk
"""

import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

with open('resources/pqdata.json', 'r') as file:
    datafull = json.load(file)
print(datafull)

# Filtering:
filtered_data = list(filter(lambda equation: len(equation[1]) >= 2, datafull))

# Splitting the data set into data and gt
data_, ground_truth = zip(*filtered_data)

# Converting input data and gt into two-dimensional NumPy arrays
data = np.array(data_)
gt = np.array(ground_truth)

print(f'Input data has shape {data.shape}.')
print(f'Groundtruth has shape {gt.shape}.')

# Defining the architecture of the neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)), # Input layer with 64 neurons and ReLU activation function
    tf.keras.layers.Dense(32, activation='relu'), # Hidden layer with 64 neurons and ReLU activation function
    tf.keras.layers.Dense(2) # Output layer with 2 neurons (for the two solutions)
])

# Displaying the model architecture
model.summary()


dataRepeated = np.tile(data, (1000,1))
gtRepeated = np.tile(gt, (1000,1))

# This is a standard way of training the network
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(dataRepeated, gtRepeated, epochs=20)

def pqFormula (pq:np.ndarray)->np.ndarray:
    '''Solves a batch of quadratic equations with the pq-formula.
    pq is a N,2 tensor.
    pq[i,0] is p and pq[i,1] is q of equation number i.
    The result is an N,2 tensor where result[i] are the two
    solutions for equation i. result[i,0] must be the smaller one.
    In case of one solution, both
    are the same. In case of no solution, they are nan.'''

    #Extract p and q from the input tensor
    p = pq[:, 0]
    q = pq[:, 1]

    #Compute the discriminant
    discriminant = (p / 2) ** 2 - q

    #Initialize the result tensor with nan
    result = np.full((pq.shape[0], 2), np.nan)

    #Find indices where there are two real solutions
    two_solutions_indices = discriminant >= 0

    #Compute the solutions for equations with two solutions
    if np.any(two_solutions_indices):
        sqrt_discriminant = np.sqrt(discriminant[two_solutions_indices])
        result[two_solutions_indices, 0] = -p[two_solutions_indices] / 2 - sqrt_discriminant
        result[two_solutions_indices, 1] = -p[two_solutions_indices] / 2 + sqrt_discriminant

    return result

testPQ = np.array([[0,-1], [-2,0], [0,1]], dtype=np.float32)
testX1X2 = np.array([[-1,1], [0, 2], [np.nan, np.nan]], dtype=np.float32)
print(pqFormula(testPQ))
print(testX1X2)

def predict (pq:np.ndarray)->np.ndarray:
    '''Runs the trained model on pq as an input batch and returns the output
    batch as a numpy 2d-array'''
    return model(pq).numpy()

def pqError (pq:np.ndarray)->np.ndarray:
    '''Computes the error of the trained model on a given input batch
    by comparing with the pq-formula.
    The error of an equation is the average error of the two solutions.
    Returns an array of errors, one number for every equation.
    '''
    
    # Run the model on the input batch
    predictions = predict(pq)

    #Calculate the solutions with the pq formula
    solutions = pqFormula(pq)

    #Calculate the error for each equation
    errors = np.abs(predictions - solutions)

    #The error is the average of the errors of the two solutions
    errors = np.mean(errors, axis=1)

    return errors

print(np.average(pqError(data))) 

def evaluate (pRange, gridSize=100):
    '''Evaluates model vs. pqFormula in the range of +/-pRange with
    gridSize grid points and returns the error as a 2d array with
    dimensions p and q. '''
    pTest = np.linspace(-pRange, pRange, gridSize)
    qTest = np.linspace(-pRange, pRange, gridSize)
    pqTest = np.stack (np.meshgrid(pTest, qTest), axis=-1)
    print(pqTest.shape)
    err = np.reshape(pqError(np.reshape(pqTest, (-1,2))), pqTest.shape[:-1])
    return err

pRange = 25
err = evaluate (pRange)
plt.imshow(err, extent=[-pRange,pRange,-pRange,pRange], origin='lower')
plt.colorbar()
plt.xlim(-pRange,pRange)
plt.ylim(-pRange,pRange)
plt.xlabel('p')
plt.ylabel('q')
plt.scatter(data[:,0], data[:,1], color='red',marker='+')
plt.title("Fehler der gelernten Nullstelle von $x^2+px+q=0$")
