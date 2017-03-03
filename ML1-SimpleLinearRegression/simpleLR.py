# Created by Qixun Qu
# quqixun@gmail.com
# 2017/01/23
# http://quqixun.com/?p=624

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_data(path):
    # This function is used to load data from text file.
    # Retuen feature matrix and label matrix respectively.
    data = pd.read_csv(path, header=None)

    # print(data.head())
    # print(data.describe())

    cols = data.shape[1]
    X = data.iloc[:, 0:cols - 1]
    y = data.iloc[:, cols - 1:cols]

    X = np.matrix(X.values)
    y = np.matrix(y.values)

    # Add a column of 1.
    X = np.insert(X, 0, 1, axis=1)

    return X, y


def plot_data(X, y):
    plt.figure()
    plt.plot(X[:,1], y, 'g.', ms=10.0)
    plt.xlim(3, 27)
    plt.xlabel('Population of City in 10,000s', fontsize=18)
    plt.ylabel('Profit in $10,000s', fontsize=18)
    plt.tick_params(labelsize=15)
    plt.grid(True)
    plt.show()


def cost_function(X, y, theta):
    # Compute errors between estimate and real values.
    errors = (X * theta) - y
    # Calculate cost function value.
    cost = sum(np.power(errors, 2)) / (2 * len(y))
    return cost


def gradient_descent(X, y, theta, alpha, iteration):
    # Initialize a vector to store all cost values.
    cost_all = np.zeros(iteration)

    for i in range(iteration):
        # Update parameters.
        theta = theta - alpha / len(y) * (X.T * (X * theta - y))
        cost_all[i] = cost_function(X, y, theta)

    return theta, cost_all


def plot_result(X, y, theta):
    # Compute two points to draw a line.
    x = np.array([4, 25])
    results = theta[1, 0] * x + theta[0, 0]

    plt.figure()
    plt.plot(x, results, 'r-', lw=2, label='Prediction')
    plt.plot(X[:, 1], y, 'g.', ms=10.0, label='Training Data')
    plt.legend(loc=2)
    plt.xlim(3, 27)
    plt.ylim(-5, 30)
    plt.xlabel('Population of City in 10,000s', fontsize=18)
    plt.ylabel('Profit in $10,000s', fontsize=18)
    plt.tick_params(labelsize=15)
    plt.grid(True)
    plt.show()


def plot_cost(cost_all, iteration):
    # Generate x -coordinate.
    iters = np.arange(0, iteration, 1)

    plt.figure()
    plt.plot(iters, cost_all, lw=2)
    # plt.ylim(4.4, 6)
    plt.xlabel('Iteration, alpha = 0.025', fontsize=18)
    plt.ylabel('Cost', fontsize=18)
    plt.tick_params(labelsize=15)
    plt.grid(True)
    plt.show()


# A test for simple linear regression.
# Only ione feature in the dataset.
if __name__ == '__main__':
    # Load data from path below.
    # Obtain features and labes.
    # Show data in a plot.
    path = os.getcwd() + '/ex1data1.txt'
    X, y = read_data(path)
    plot_data(X, y)

    # Set argument:
    # step size, the number of iteration,
    # initial function parameters.
    alpha = 0.025
    iteration = 1500

    theta_num = X.shape[1]
    theta = np.matrix(np.zeros((theta_num, 1)))

    # Implement gradient descent method to
    # estimate function parameters (theta).
    theta, cost_all = gradient_descent(X, y, theta, alpha, iteration)

    # Plot a line with paramenters of theta.
    plot_result(X, y, theta)

    # Plot values of cost function w.r.t. iteration.
    plot_cost(cost_all, iteration)
