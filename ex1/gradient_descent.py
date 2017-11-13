# -*- coding: utf-8 -*-

import computeCost as cc
import numpy as np

def gradientDescent(X, y, theta, alpha, iters):
    m = len(y)
    J_history = np.zeros(iters)
    """
    for iter in range(iters):
        h = X @ theta.T
        theta = theta - alpha / m * (X.T @ (h - y))
        J_history[iter]= cc.computeCost(X, y, theta)
"""
    return J_history
