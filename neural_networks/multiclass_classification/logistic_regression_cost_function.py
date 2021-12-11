#!/usr/bin/env python3
import numpy as np
import math


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-1.0 * z))


def calc_hypothesis(theta, x):
    z = np.dot(theta, x)
    return sigmoid(z)

def calc_logistic_regression_cost_per_row(y, h):
    return -1 * y * math.log(h) - (1 - y) * log( 1 - h)


def logistic_regression_cost_func(theta, X, Y, lambda_param):
    """
    Computes the cost of using theta as the parameter for regularized logistic
    regression and the gradient of the cost w.r.t. to the parameters.
    """
    (num_samples, num_features) = X.shape

    factor = 1.0 / num_samples

    hypothesis_arr = list(map(sigmoid, X @ theta.reshape(num_features, 1)))
    first_sum_arr = np.multiply(Y, list(map(math.log, hypothesis_arr)))
    second_sum_factor = np.ones(num_samples) - Y
    second_sum_log = np.ones(num_samples) - hypothesis_arr
    second_sum_arr = np.multiply(second_sum_factor, list(map(math.log, second_sum_log)))
    cost_arr = -1 * first_sum_arr - second_sum_arr

    theta_arr = theta[1:num_features]
    regularization_factor = lambda_param / (2.0 * num_samples)
    regularization_term = regularization_factor * np.sum(np.multiply(theta_arr, theta_arr))

    return factor * np.sum(cost_arr) + regularization_term

def logistic_regression_gradient(theta, X, Y, lambda_param):
   pass
