#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: omerfaruk

There are many points in a plane belonging to different classes (see graphic below).
To differentiate the classes based on the positions of the points. 
To do this, it's required to analyze how close they come to each other and calculate,
for each pair of classes, the smallest distance between two points belonging to each class.
"""

import math
import random
import tensorflow as tf
import matplotlib.pyplot as plt

# Make data
nClass = 8
nSamples = 100
noise = 0.1
data = tf.constant([[
    [math.cos(j/nClass*2*math.pi)+random.gauss(0,noise), math.sin(j/nClass*2*math.pi)+random.gauss(0,noise)] 
    for k in range(nSamples)] for j in range(nClass)], dtype=tf.float32)
approxDistances = tf.constant([
    [max(0, 2*math.sin(abs(i-j)*math.pi/nClass)-4*noise) for i in range(nClass)] 
                               for j in range(nClass)])
print(f'data shape is class * samples * coordinates {data.shape}')

# Visualize data
plt.axis('equal')
for j in range(len(data)):
    plt.scatter(data[j,:,0], data[j,:,1], s=0.5, label=f'class {j}')
plt.legend()
plt.show()

def smallest_distances(data):
    '''data is a classes*points*coordinates tensor. The function returns 
    a two-dimensional tensor, where result[i,j] is the smallest
    distance between points of class i and j.'''
    
    # Expand dimensions for broadcasting
    data_expanded_i = tf.expand_dims(data, 1)
    data_expanded_j = tf.expand_dims(data, 0)

    # Compute differences
    diffs = data_expanded_i - data_expanded_j

    # Compute Euclidean distances
    distances = tf.norm(diffs, axis=-1)

    # Find minimum distances
    smallest_dists = tf.reduce_min(distances, axis=-1)
    
    return smallest_dists

result = smallest_distances(data)
print(f'result\n{result}\n')
print(f'average difference {tf.reduce_mean(tf.abs(result-approxDistances))}')