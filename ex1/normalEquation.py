# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
"""
dataset = pd.read_csv('ex1data2.txt', header=None)
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2:3].values
X = X.astype('float')
y = y.astype('float')
"""
dataset = pd.read_csv('ex1data1.txt', header=None)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1:2].values

ones = np.ones(len(X))

X = np.column_stack((ones, X))


theta_res = np.array([[-3.630291439404360165], [1.166362350335581999]])


theta_normal = np.linalg.pinv(X.T @ X) @ X.T @ y

y_pred = X @ theta_res
y_pred2 = X @ theta_normal

plt.scatter(X[:, 1], y)
plt.plot(X[:, 1], y_pred)
plt.plot(X[:, 1], y_pred2)