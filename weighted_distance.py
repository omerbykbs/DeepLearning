#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: omerfaruk
"""

import numpy as np

def weighted_distance (x,y,W):
    '''Returns the W-weighted distance from x to y. x and y are n dimensional
    vectors, W is an n*n matrix. x, y, W can have arbitrary outer batch 
    dimensions.'''
    # Answer:
    diff = x - y
    difference_first = np.einsum('...i,...ij->...j', diff, W)
    difference_final = np.einsum('...i,...i->...', difference_first, diff)
    return np.sqrt(difference_final)
