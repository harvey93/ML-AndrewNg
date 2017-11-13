# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv('ex1data2.txt', header=None)
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2:3].values
X = X.astype('float')
y = y.astype('float')


# Feature Scaling
X[:, 0] = (X[:, 0] - np.average(X[:, 0])) / np.std(X[:, 0])
X[:, 1] = X[:, 1] - np.average(X[:, 1]) / np.std(X[:, 1])


ones = np.ones(len(X))
X = np.column_stack((ones, X))

theta = np.zeros((3,1))
iterations = 1500
alpha = 0.1

    
def computeCost(X, y, theta):
    m = len(X)
    h = X @ theta
    J = 1/(2 * m) * np.sum((h - y) ** 2)
    return J
    

def gradientDescent(X, y, theta, alpha, iters):
    m = len(X)
    J_history = np.zeros(iters)
    
    for iter in np.arange(iters):
        h = X @ theta
        theta = theta - alpha * (1/m) * (X.T.dot(h - y))
        J_history[iter] = computeCost(X, y, theta)
    return theta, J_history
    

theta_res, J_history = gradientDescent(X, y, theta, alpha, iterations)    
    


y_pred = X @ theta_res

plt.plot(J_history)
