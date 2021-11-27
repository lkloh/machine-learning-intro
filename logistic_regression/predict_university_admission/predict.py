#!/usr/bin/env python3

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
import scipy.optimize as optimize

ALPHA = 0.01
NUM_ITERATIONS = 1500


def load_data(filename):
    file = open(filename, "r")
    contents = file.readlines()
    num_vars = len(contents)

    X = np.zeros(shape=(num_vars, 2))
    Y = np.zeros(num_vars, dtype=bool)
    for i, line in enumerate(contents):
        chunks = line.split(",")
        X[i][0] = float(chunks[0])
        X[i][1] = float(chunks[1])
        Y[i] = True if int(chunks[2]) == 1 else False
    return [X, Y]


def visualize_data(X, Y, optimized_theta=None):
    X_admitted = X[Y]

    Y_rejected = np.invert(Y)
    X_rejected = X[Y_rejected]

    fig = make_subplots(
        rows=1,
        cols=1,
        x_title="Exam 1 Score",
    )
    fig.update_layout(title="Exam 2 Score against Exam 1 Score")

    fig.append_trace(
        go.Scatter(
            x=X_admitted[:, 0],
            y=X_admitted[:, 1],
            name="Admitted",
            mode="markers",
        ),
        row=1,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            x=X_rejected[:, 0],
            y=X_rejected[:, 1],
            name="Rejected",
            mode="markers",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Exam 2 Score", row=1, col=1)

    if optimized_theta is not None:
        pass

    fig.write_html("admitted_and_rejected_visualization.html")


"""
Logistic regression hypothesis is:
h_theta(x) = g(theta^T x)
          1
g(x) = -------
            -z
       1 + e
"""


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-1.0 * z))


def calc_hypothesis(theta, x):
    z = np.dot(theta, x)
    return sigmoid(z)


"""
Compute cost for logistic regression
"""


def calc_cost_func(theta, X, Y):
    (num_samples, num_factors) = X.shape

    cost_factor = 1.0 / num_samples
    cost_sum = 0
    for sample_idx in range(num_samples):
        xi = X[sample_idx, :]
        yi = Y[sample_idx]
        h = calc_hypothesis(theta, xi)
        cost_sum -= yi * math.log(h)
        cost_sum -= (1 - yi) * math.log(1 - h)
    return cost_factor * cost_sum


"""
Compute gradient for logistic regression
"""


def calc_gradient(theta, X, Y):
    (num_samples, num_factors) = X.shape

    gradient = np.zeros(num_factors)
    gradient_factor = 1.0 / num_samples
    for factor_idx in range(num_factors):
        gradient_sum = 0
        for sample_idx in range(num_samples):
            xi = X[sample_idx, :]
            yi = Y[sample_idx]
            h = calc_hypothesis(theta, xi)
            gradient_sum += (h - yi) * xi[factor_idx]
        gradient[factor_idx] = gradient_factor * gradient_sum
    return gradient


if __name__ == "__main__":
    [X, Y] = load_data("../assignment/ex2data1.txt")
    (num_samples, num_features) = X.shape
    visualize_data(X, Y)

    # Add intercept term to X
    X = np.c_[np.ones(shape=(num_samples, 1)), X]

    print("sigmoid(-99) = %f" % sigmoid(-99))
    print("sigmoid(0) = %f" % sigmoid(0))
    print("sigmoid(99) = %f" % sigmoid(99))

    initial_theta = np.zeros(3)
    initial_J = calc_cost_func(initial_theta, X, Y)
    print("Cost from initial theta: %f" % initial_J)

    result = optimize.minimize(
        fun=calc_cost_func,
        x0=initial_theta,
        args=(X, Y),
        method="TNC",
        jac=calc_gradient,
    )
    optimal_theta = result.x
    print("optimal theta is: ", optimal_theta, ", and cost is: ", result.fun)

    visualize_data(X, Y, optimal_theta)