# Created by Qixun Qu
# quqixun@gmail.com
# 2017/03/25
#

import os
import numpy as np
import scipy.io as sio
import scipy.optimize as sopt


def read_data(path):
    # Read data from given .mat file
    data = sio.loadmat(path)

    # Extract variables from data
    # by their names
    X = data['X']
    y = data['y']

    # Insert one column of 1 before X
    X = np.insert(X, 0, 1, axis=1)

    return X, y


def sigmoid(z):
    # The sigmoid function
    return 1 / (1 + np.exp(-z))


def rand_initialize_weights(L_in, L_out):
    epsilon_init = 0.12
    return np.random.rand(L_out, L_in + 1) * 2 * epsilon_init - epsilon_init


def reshape_params(nn_params, input_layer_size,
                   hidden_layer_size, num_labels):
    theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        [hidden_layer_size, input_layer_size + 1])
    theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        [num_labels, hidden_layer_size + 1])

    return theta1, theta2


def sub2ind(shape, rows, cols):
    return rows * shape[1] + cols


def one_hidden_layer_network(theta1, theta2, X, y, num_labels):
    z2 = X.dot(theta1.T)
    a2 = np.insert(sigmoid(z2), 0, 1, axis=1)
    h_theta = sigmoid(a2.dot(theta2.T))

    m = X.shape[0]
    shape = [m, num_labels]
    yc = np.zeros(shape).flatten()
    yc[sub2ind(shape, np.arange(m), y.T - 1)] = 1

    return h_theta, np.reshape(yc, shape), z2, a2


def cost(nn_params, X, y, input_layer_size,
         hidden_layer_size, num_labels, ld=1):
    theta1, theta2 = reshape_params(nn_params, input_layer_size,
                                    hidden_layer_size, num_labels)
    m = len(y)

    h_theta, yc, _, _ = one_hidden_layer_network(theta1, theta2,
                                                 X, y, num_labels)

    cost = \
        -np.sum(np.multiply(yc, np.log(h_theta)) +
                np.multiply((1 - yc), np.log(1 - h_theta))) / m + \
        ld / (2 * m) * (np.sum(np.power(theta1[:, 1:], 2)) +
                        np.sum(np.power(theta2[:, 1:], 2)))

    return cost


def sigmoidGradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


def gradient(nn_params, X, y, input_layer_size,
             hidden_layer_size, num_labels, ld=1):
    theta1, theta2 = reshape_params(nn_params, input_layer_size,
                                    hidden_layer_size, num_labels)
    m = len(y)

    h_theta, yc, z2, a2 = one_hidden_layer_network(
        theta1, theta2, X, y, num_labels)

    delta3 = h_theta - yc
    delta2 = np.multiply(delta3.dot(theta2[:, 1:]), sigmoidGradient(z2))

    Delta1 = delta2.T.dot(X)
    Delta2 = delta3.T.dot(a2)

    theta1_grad = Delta1 / m
    theta2_grad = Delta2 / m

    theta1_grad[:, 1:] += ld / m * theta1[:, 1:]
    theta2_grad[:, 1:] += ld / m * theta2[:, 1:]

    grad = np.concatenate((theta1_grad.flatten(),
                           theta2_grad.flatten()), axis=0)

    return grad


def predict(theta1, theta2, X):
    h1 = sigmoid(X.dot(theta1.T))
    h1 = np.insert(h1, 0, 1, axis=1)
    h2 = sigmoid(h1.dot(theta2.T))

    pred = np.argmax(h2, axis=1) + 1

    return pred.reshape([X.shape[0], 1])


if __name__ == '__main__':
    # Load training data
    path = os.getcwd() + '/ex4data1.mat'
    X, y = read_data(path)

    # Setup parameters you will use
    # 20x20 input image of digits
    input_layer_size = 400
    # 25 hidden neurons
    hidden_layer_size = 25
    # 10 labels from 1 to 10,
    # note that we have mapped "0" to label 10
    num_labels = 10
    # regularization parameter
    ld = 1

    # Randomly initialize parameters
    init_theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
    init_theta2 = rand_initialize_weights(hidden_layer_size, num_labels)

    initial_nn_params = np.concatenate((init_theta1.flatten(),
                                        init_theta2.flatten()),
                                       axis=0)

    #
    estimate = sopt.fmin_tnc(func=cost, x0=initial_nn_params, fprime=gradient,
                             args=(X, y, input_layer_size, hidden_layer_size,
                                   num_labels, ld), maxfun=200)

    theta1, theta2 = reshape_params(estimate[0], input_layer_size,
                                    hidden_layer_size, num_labels)

    pred = predict(theta1, theta2, X)
    print("Training accuracy: {0:.2f}%".format(np.mean((pred == y) * 1) * 100))
