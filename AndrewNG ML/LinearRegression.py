import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
##Linear Regression with one variable

data = pd.read_csv('.\ex1\ex1data1.txt', header=None)
data.columns = ['Population', 'Profit']
X = data.iloc[:, 0].values #first column i.e. Population
y = data.iloc[:, 1].values #second column i.e. Profit of food truck

m = len(y)

# plt.scatter(X, y)
# plt.title('Population vs Food Truck Profit')
# plt.xlabel('Population in 10,000s')
# plt.ylabel('Profit in $10,000s')
# plt.show()

ones = np.ones((m, 1)) #X0 initialized to ones
theta = np.zeros((2, 1)) #Theta initialized to (0, 0)

X = np.append(ones, X.reshape(m, 1), axis = 1)
y = y.reshape(m, 1)

#print(X.shape, y.shape)

def computeCost(X, y, theta):
    m = len(y)
    error = np.dot(X, theta) - y
    return np.sum(np.power(error, 2)) /(2*m)
print("Value of cost when Theta = (0, 0) is :", computeCost(X, y, theta))


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_errors = []

    for i in range(num_iters):
        pred = X.dot(theta)
        error = np.dot(X.T, (pred - y))
        descent = alpha*(1/m)*error
        theta-= descent
        J_errors.append(computeCost(X, y, theta))
    return theta, J_errors

theta, J = gradientDescent(X, y, theta, 0.01, 1500)

print("h(x) = " + str(round(theta[0,0], 2)) + " + " + str(round(theta[1,0], 2)) + "x1")


# plt.plot(J)
# plt.xlabel('Number of iterations')
# plt.ylabel("Cost/Error")
# plt.title("Gradient Descent over number of iterations")
# plt.show()


# plt.scatter(data[['Population']], data[['Profit']])
# plt.plot(X[:,1], np.dot(X, theta), color = 'r')
# plt.xticks(np.arange(5,30,step=5))
# plt.yticks(np.arange(-5,30,step=5))
# plt.xlabel("Population of City (10,000s)")
# plt.ylabel("Profit ($10,000")
# plt.title("Profit vs Population")
# plt.show()


def predict(X, theta):
    pred = np.dot(X, theta)
    print(pred[0])

predict(np.array([1,7]), theta)