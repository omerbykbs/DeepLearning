#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: omerfaruk
"""

import numpy as np
import tensorflow as tf
if len(tf.config.list_physical_devices('GPU')):
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

x = np.array([-1, 2, 1, 3], dtype=float)
y = np.array([4, 1, 0, 4], dtype=float)
a = np.array([2, 0, 1], dtype=float)
eta = 0.001

# Computing y_hat
x_powers = tf.stack([x**j for j in range(3)], axis=-1)
y_hat = tf.reduce_sum(a * x_powers, axis=-1)

# Computing Loss
L = tf.reduce_sum((y_hat - y)**2)
print("Loss:", L.numpy())

# Computing the derivative of L with respect to y_hat
grad_y_hat = 2 * (y_hat - y)
print('dL/dy_hat: ',grad_y_hat)

# Computing the derivative of y_hat with respect to a
grad_a_y_hat = x_powers
print('dy_hat/da: ', grad_a_y_hat)

# Computing the derivative of L with respect to a using the chain rule
grad_a = tf.reduce_sum(grad_y_hat[:, None] * grad_a_y_hat, axis=0)
print("Gradients:", grad_a.numpy())

# Step 5: Update parameters
a -= eta * grad_a

# Step 6: Compute new y_hat and Loss
y_hat_new = tf.reduce_sum(a * x_powers, axis=-1)
L_new = tf.reduce_sum((y_hat_new - y)**2)
print("New Loss:", L_new.numpy())
print("New Parameters:", a.numpy())

# Step 7: Verify that the new loss is less than the original loss
assert L_new < L