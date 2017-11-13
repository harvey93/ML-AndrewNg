# -*- coding: utf-8 -*-
import numpy as np

def computeCost(X, y, theta):
    inner = np.power(((X @ theta.T)) - y, 2)
    return np.sum(inner) / (2 * len(X))

   

#print(computeCost(1,2,3))    