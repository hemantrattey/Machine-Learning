import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('.\ex1\ex1data2.txt', header = None)
data.columns = ['Size', 'Number of Bedrooms', 'Price']

# print(data.describe())

X2 = data.iloc[:, 0:2].values
y2 = data.iloc[:, 2].values

X2 = X = (X2 - np.mean(X2))/np.std(X2)
m2 = len(y2)

ones = np.ones((m2, 1))
# print(X.shape)
X2 = np.append(ones, X2.reshape(m2, 2), axis = 1) #append a column of 1s for X0 hence 3 columns for X0, X1 and X2
# print(X.shape)
y2 = y2.reshape(m2,1)

theta2 = np.zeros((3, 1))

def computeCostMulti(X, y, theta):
    m = len(y)
    error = np.dot(X, theta) - y
    return np.sum(np.power(error, 2)) / (2*m)

# print(computeCostMulti(X2, y2, theta2))

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = len(y)
    for i in range (num_iters):
        temp = np.dot(X, theta) - y
        temp = np.dot(X.T, temp) 
        theta = theta - (alpha/m) * temp
    return theta
theta = gradientDescentMulti(X2, y2, theta2, 0.01, 400)
print(theta)