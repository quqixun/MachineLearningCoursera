# Created by Qixun Qu
# quqixun@gmail.com
# 2017/03/21
# http://quqixun.com/?p=753

import os
import numpy as np
import scipy.io as sio


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


def read_weights(path):
    # Read weights form given .mat file
    weights = sio.loadmat(path)

    # The first weight is for the hidden layer
    # The second weight is for the output layer
    theta1 = weights['Theta1']
    theta2 = weights['Theta2']

    return theta1, theta2


def sigmoid(z):
    # The sigmoid function
    return 1 / (1 + np.exp(-z))


def predict(theta1, theta2, X):
    # Compute the output for the hidden layer
    # Then inset the bias item
    a2 = sigmoid(X.dot(theta1.T))
    a2 = np.insert(a2, 0, 1, axis=1)

    # Compute the values of output layer
    pp = sigmoid(a2.dot(theta2.T))
    p = np.argmax(pp, axis=1) + 1

    return np.reshape(p, [X.shape[0], 1])


if __name__ == '__main__':
    # Load data from .mat file
    data_path = os.getcwd() + '/ex3data1.mat'
    X, y = read_mat(data_path)

    # Load weights from .mat file
    weights_path = os.getcwd() + '/ex3weights.mat'
    theta1, theta2 = read_weights(weights_path)

    # Do prediction on training set
    #　And compute training accuracy
    pred = predict(theta1, theta2, X)
    accuracy = np.mean((y == pred) * 1) * 100

    print('Training set accuracy: {0:.2f}%'.format(accuracy))
    # The accuracy should be 97.52%.

    # Print some cases with its prediction and real class
    for i in range(0, 5000, 500):
        pred = predict(theta1, theta2, np.reshape(X[i], [1, X.shape[1]]))
        print('The prediction is {}, it should be {}.'.format(pred[0], y[i]))
