#!/usr/bin/env python3
import numpy as np
import math
import compute_cost


def calc_gradient_descent(X, y, theta, alpha, num_iterations):
    # num training examples
    m = len(y)

    # num feature variables
    n = len(theta)

    cost_function_history = np.zeros(num_iterations)

    for iteration_count in range(num_iterations):

        # Update theta
        for feature_idx in range(n):
            sum = 0
            for training_idx in range(m):
                xi = X[training_idx]
                yi = y[training_idx]
                sum += (compute_cost.calc_hypothesis(xi, theta) - yi) * xi

            factor = alpha * (1.0 / m)
            theta[feature_idx] = theta[feature_idx] - factor * sum

        cost_function_history[iteration_count] = compute_cost.calc_cost(X, y, theta)

    return [cost_function_history, theta]
