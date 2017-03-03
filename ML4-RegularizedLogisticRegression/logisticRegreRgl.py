# Made by Qixun QU
# quqixun@gmail.com
# 2017/03/03
# http://quqixun.com/?p=736

import os
import numpy as np
import pandas as pd
import scipy.optimize as sopt
import matplotlib.pyplot as plt


def read_data(path):
    # This function is used to load data from text file.
    # Retuen feature matrix and label matrix respectively.
    data = pd.read_csv(path, header=None)

    cols = data.shape[1]
    X = data.iloc[:, 0:cols - 1]
    y = data.iloc[:, cols - 1:cols]

    X = np.matrix(X.values)
    y = np.matrix(y.values)

    # Add a column of 1.
    X = np.insert(X, 0, 1, axis=1)

    return X, y


def plot_data(X, y):
    # Obtain index of samples with label 1
    lb1_idx = np.where(y == 1)[0]
    # Obtain index of samples with label 0
    lb0_idx = np.where(y == 0)[0]

    plt.figure()

    # Plot all points of two groups
    plt.plot(X[lb1_idx, 1], X[lb1_idx, 2], 'g.', ms=10.0, label='y = 1')
    plt.plot(X[lb0_idx, 1], X[lb0_idx, 2], 'r.', ms=10.0, label='y = 0')

    plt.xlabel('Microchip Test 1', fontsize=18)
    plt.ylabel('Microchip Test 2', fontsize=18)
    plt.tick_params(labelsize=15)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


def map_features(X1, X2):
    # Generate 28 new features based on
    # two given features
    degree = 6

    if type(X1) == np.float64:
        map_X = np.ones(1)
    else:
        map_X = np.ones([X1.shape[0], 1])

    for i in range(1, degree + 1):
        for j in range(i + 1):
            temp = np.multiply(np.power(X1, i - j), np.power(X2, j))
            map_X = np.hstack([map_X, temp])

    return map_X


def sigmoid(z):
    # Complete sigmoid function
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y, ld=1):
    # Compute cost function
    # Convert theta into a 3x1 matrix
    theta = np.asmatrix(theta)
    theta = theta.reshape([X.shape[1], 1])

    # Calculate regularised cost function
    cost = -(y.T * np.log(sigmoid(X * theta)) + (1 - y).T * np.log(1 - sigmoid(X * theta))) / len(y)

    cost += (theta[1:].T * theta[1:]) * ld / (2 * len(y))

    return cost


def gradient(theta, X, y, ld=1):
    # Compute gradient
    # Convert theta into a 3x1 matrix
    theta = np.asmatrix(theta)
    theta = theta.reshape([X.shape[1], 1])

    # Calculate regularised gradient
    grad = X.T * (sigmoid(X * theta) - y) / len(y)

    grad[1:] = grad[1:] + ld / len(y) * theta[1:]

    return grad


def plot_result(X, y, theta, num):
    # Compute points to draw a contour
    theta = np.reshape(theta, [num, 1])
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)

    z = np.zeros([len(u), len(v)])
    for i in range(len(u)):
        for j in range(len(v)):
            map_f = np.reshape(map_features(u[i], v[j]), [1, num])
            z[i, j] = np.dot(map_f, theta)

    # Obtain index of samples with label 1
    lb1_idx = np.where(y == 1)[0]
    # Obtain index of samples with label 0
    lb0_idx = np.where(y == 0)[0]

    plt.figure()

    # Plot all points of two groups
    plt.plot(X[lb1_idx, 1], X[lb1_idx, 2], 'g.', ms=10.0, label='y = 1')
    plt.plot(X[lb0_idx, 1], X[lb0_idx, 2], 'r.', ms=10.0, label='y = 0')

    # Plot decision boundary
    plt.contour(u, v, z, 1, linewidths=2, colors='k')

    plt.title('Decision Boundary', fontsize=20)
    plt.xlabel('Microchip Test 1', fontsize=18)
    plt.ylabel('Microchip Test 2', fontsize=18)
    plt.tick_params(labelsize=15)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Read and plot data from file
    path = os.getcwd() + '/ex2data2.txt'
    X, y = read_data(path)
    plot_data(X, y)

    # Generate 28 new features
    map_X = map_features(X[:, 1], X[:, 2])

    # Initialize theta
    theta = np.zeros([map_X.shape[1], 1])

    # Set the value for lambda
    ld = 1

    # Find out the local minimum of cost function
    estimate = sopt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(map_X, y, ld))

    # Calculate cost function by applying estimated theta
    cost = cost(estimate[0], map_X, y)

    print("Estimated parameters are {},\nnow the value of cost function is {}."
          .format(str(estimate[0]).strip('[]'), str(cost).strip('[]')))

    # Plot decision boundary
    plot_result(X, y, estimate[0], map_X.shape[1])
