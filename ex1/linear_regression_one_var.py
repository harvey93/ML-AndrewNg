# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv('ex1data1.txt', header=None)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1:2].values

ones = np.ones(len(X))

X = np.column_stack((ones, X))
theta = np.zeros((2,1))
iterations = 1500
alpha = 0.01

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
    return(theta, J_history)

[theta_res, J_hist] = gradientDescent(X, y, theta, alpha, iterations)

#plt.plot(J_hist)
#plt.ylabel("Cost function")
#plt.xlabel("Iterations")

y_pred = X @ theta_res

plt.xlabel("Population of City in 10,000")
plt.ylabel("Profits in $10,000")
plt.title("Marketplace")
plt.scatter(X[:, 1], y)
plt.plot(X[:, 1], y_pred)

