# Create by Qixun Qu
# quqixun@gmail.com
# 2017/03/20
# http://quqixun.com/?p=753

import os
import numpy as np
import scipy.io as sio
import scipy.optimize as sopt


def read_mat(path):
    # Read data form given .mat file
    # use the function built in the scipy
    data = sio.loadmat(path)

    # Extract variables form data
    # by their names
    X = data['X']
    y = data['y']

    # Insert one column of 1 before X
    X = np.insert(X, 0, 1, axis=1)

    return X, y


def sigmoid(z):
	# The sigmoid function
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y, ld=1):
	# Compute the cost function
	# Convert theta to a vector
    theta = np.asmatrix(theta)
    theta = theta.reshape([X.shape[1], 1])

    # Compute cost function
    # The first parameter in theta has no necessary to be regulized
    cost = -(y.T * np.log(sigmoid(X * theta)) + (1 - y).T * np.log(1 - sigmoid(X * theta))) / len(y)
    cost += (theta[1:].T * theta[1:]) * ld / (2 * len(y))

    return cost


def gradient(theta, X, y, ld=1):
	# Compute the gradient in each iteration
	# Convert theta to a vector
    theta = np.asmatrix(theta)
    theta = theta.reshape([X.shape[1], 1])

    # Compute gradient of each parameter except the first one
    grad = X.T * (sigmoid(X * theta) - y) / len(y)
    grad[1:] = grad[1:] + ld / len(y) * theta[1:]

    return grad


def oneVsAll(X, y, num, ld=1):
	# Compute parameters for each class
	# Get the number of columns of feature matrix,
	# i.e. the number of features
    n = X.shape[1]

    # Initialize all parameters
    # For each class of data, corresponding parameters
    # are stotred in one row, the number of rows euqals
    # to the number of types of labels
    all_theta = np.zeros([num_labels, n])

    # Initialize one set of parameters for each type of data
    initial_theta = np.zeros([n, 1])

    for c in range(1, num_labels + 1):
    	# In each iteration, set one type of labels to 1,
    	# the other labels to 0, here is an example:
    	# All labels:     1 2 3 4 5 6 7 8 9 10
    	# 1st iteration:  1 0 0 0 0 0 0 0 0 0
    	#                 estimate parameters for 1st type of data
    	# 2nd iteration:  0 1 0 0 0 0 0 0 0 0
    	#                 estimate parameters for 2nd type of data
    	# ...
    	# 10th iteration: 0 0 0 0 0 0 0 0 0 1
    	#                 estimate parameters for 10th type of data
    	# All parameters form the matrix all_theta
        y_c = (y == c) * 1
        estimate = sopt.fmin_tnc(func=cost, x0=initial_theta, fprime=gradient, args=(X, y_c, ld), maxfun=500)
        all_theta[c - 1] = estimate[0]

    return all_theta


def predictOneVsAll(theta, X):
	# Use sigmoid function as the linear classifier
	# to do the prediction
	# The output of sigmoid function has the same
	# number of columns as the classes of data
	# For instance, the output pp may look like as:
	# 0.05 0 0 0 0 0.1 0.05 0.8 0 0
	# In this case, the biggest probability locates at
	# 8th column, which means this data is regarded as
	# the data in 8th class
    pp = sigmoid(X.dot(theta.T))
    p = np.argmax(pp, axis=1) + 1

    return p.reshape([X.shape[0], 1])


if __name__ == '__main__':
	# Read data form .mat file
    path = os.getcwd() + '/ex3data1.mat'
    X, y = read_mat(path)

    # Set the number of classes of data
    # In this case, there are 10 classes,
    # 1 for digit 1, 2 for digit 2, ...,
    # 9 for digit 9, 10 for digit 0
    num_labels = 10

    # Set the parameter for regulization
    ld = 0.1

    # Estimate parameter matrix
    all_theta = oneVsAll(X, y, num_labels, ld)

    # Do the prediction for training data
    pred = predictOneVsAll(all_theta, X)

    # Calculate the training accuracy
    accuracy = np.mean((pred == y) * 1) * 100
    print('The prediction accuracy is {0:.2f}%.'.format(accuracy))
    # The accuracy should be 96.46%.
