#!/usr/bin/env python3
import numpy as np
import math


def sigmoid(z):
    return 1.0 / (1.0 + math.log(-1.0 * z))


def calc_hypothesis(theta, x):
    z = np.dot(theta, x)
    return sigmoid(z)


def logistic_regression_cost_func(theta, X, Y, lambda_param):
    """
    Computes the cost of using theta as the parameter for regularized logistic
    regression and the gradient of the cost w.r.t. to the parameters.
    """
    (num_samples, num_features) = X.shape

    factor = 1.0 / num_samples

    sum = 0
    for sample_idx in range(num_samples):
        yi = Y[sample_idx]
        xi = X[sample_idx, :]

        h = calc_hypothesis(theta, xi)
        sum -= yi * math.log(h)
        sum -= (1 - yi) * math.log(1.0 - h)

    return factor * sum
