#!/usr/bin/env python3

import numpy as np
from logistic_regression_cost_function import (
    logistic_regression_cost_func,
    logistic_regression_gradient,
)
import scipy.optimize as optimize

MAX_ITERATIONS = 50

def one_vs_all_classifier(X, Y, num_labels, lambda_param):
    """
    Trains multiple logistic regression classifiers.

    Returns all the classifiers in a matrix `all_theta`, where the i-th row of `all_theta`
    corresponds to the classifier for label i
    """
    [num_samples, num_features] = X.shape

    all_theta = np.zeros(shape=(num_labels, num_features + 1))

    # Add intercept term to X
    X = np.c_[np.ones(shape=(num_samples, 1)), X]

    for i in range(num_labels):
        num = 10 if i == 0 else num
        y_labeled_data = (Y == num).astype(int)

        theta = np.zeros(num_features + 1)
        result = optimize.minimize(
            fun=logistic_regression_cost_func,
            x0=theta,
            args=(X, y_labeled_data, lambda_param),
            method="TNC",
            jac=logistic_regression_gradient,
            options={"maxiter": MAX_ITERATIONS},
        )
        optimal_theta = result.x
        all_theta[i, :] = optimal_theta

    return all_theta
