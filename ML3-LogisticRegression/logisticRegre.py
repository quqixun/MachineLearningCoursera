# Created by Qixun Qu
# quqixun@gmail.com
# 2017/02/18
# http://quqixun.com/?p=720

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

    plt.xlabel('Exam 1 Score', fontsize=18)
    plt.ylabel('Exam 2 Score', fontsize=18)
    plt.tick_params(labelsize=15)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


def sigmoid(z):
    # Complete sigmoid function
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y):
    # Compute cost function
    # Convert theta into a 3x1 matrix
    theta = np.asmatrix(theta)
    theta = theta.reshape([3, 1])

    # Calculate cost function as
    # cost = -(y*log(h(X*theta)) + (1-y)*log(1-h(X*theta))) / m
    cost = -(y.T * np.log(sigmoid(X * theta)) + (1 - y).T * np.log(1 - sigmoid(X * theta))) / len(y)

    return cost


def gradient(theta, X, y):
    # Compute gradient
    # Convert theta into a 3x1 matrix
    theta = np.asmatrix(theta)
    theta = theta.reshape([3, 1])

    # Calculate gradient as
    # grad = X * (h(X*theta) - y) / m
    grad = X.T * (sigmoid(X * theta) - y) / len(y)
    return grad


def plot_result(X, y, theta):
    # Compute two points to draw a line
    x = np.array([30, 92])
    results = -1 / theta[2] * (theta[1] * x + theta[0])

    # Obtain index of samples with label 1
    lb1_idx = np.where(y == 1)[0]
    # Obtain index of samples with label 0
    lb0_idx = np.where(y == 0)[0]

    plt.figure()
    # Plot decision boundary
    plt.plot(x, results, 'k-', lw=2)

    # Plot all points of two groups
    plt.plot(X[lb1_idx, 1], X[lb1_idx, 2], 'g.', ms=10.0, label='y = 1')
    plt.plot(X[lb0_idx, 1], X[lb0_idx, 2], 'r.', ms=10.0, label='y = 0')

    plt.title('Decision Boundary', fontsize=20)
    plt.xlabel('Exam 1 Score', fontsize=18)
    plt.ylabel('Exam 2 Score', fontsize=18)
    plt.tick_params(labelsize=15)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Read and plot data from file
    path = os.getcwd() + '/ex2data1.txt'
    X, y = read_data(path)
    plot_data(X, y)

    # Initialize theta
    theta = np.zeros([X.shape[1], 1])

    # Estimate theta by optimized method
    estimate = sopt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
    # Calculate cost function by applying estimated theta
    cost = cost(estimate[0], X, y)

    print("Estimated parameters are {},\nnow the value of cost function is {}."
          .format(str(estimate[0]).strip('[]'), str(cost).strip('[]')))

    # Plot decision boundary
    plot_result(X, y, estimate[0])
