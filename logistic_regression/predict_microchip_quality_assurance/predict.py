#!/usr/bin/env python3

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
import scipy.optimize as optimize

NUM_ITERATIONS = 400
LAMBDA_PARAM = 1
MAP_DEGREE = 6


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


def map_feature_helper(x1, x2):
    num_features = 28
    output = np.ones(num_features)
    feature_idx = 1
    for i in range(1, MAP_DEGREE + 1):
        for j in range(0, i + 1):
            r = math.pow(x1, i - j) * math.pow(x2, j)
            output[feature_idx] = r
            feature_idx += 1
    return output


def visualize_data_with_decision_boundary(X, Y, optimized_theta):
    X_admitted = X[Y]

    Y_rejected = np.invert(Y)
    X_rejected = X[Y_rejected]

    min_test1 = min(X[:, 0])
    max_test1 = max(X[:, 0])

    min_test2 = min(X[:, 1])
    max_test2 = max(X[:, 1])

    u = np.linspace(min_test1, max_test1, 50)
    v = np.linspace(min_test2, max_test2, 50)

    z = np.zeros(shape=(len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            mapped_x = map_feature_helper(u[i], v[j])
            z[i][j] = np.dot(mapped_x, optimized_theta)

    fig2 = go.Figure(
        data=go.Contour(
            z=z,
            x=u,
            y=v,
            contours_coloring="lines",
            line_width=2,
            contours=dict(
                start=0,
                end=0,
            ),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=X_admitted[:, 0], y=X_admitted[:, 1], mode="markers", name="Admitted"
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=X_rejected[:, 0], y=X_rejected[:, 1], mode="markers", name="Rejected"
        )
    )

    fig2.write_html("decision_boundary_microchip.html")


def visualize_data(X, Y):
    X_admitted = X[Y]

    Y_rejected = np.invert(Y)
    X_rejected = X[Y_rejected]

    fig = make_subplots(
        rows=1,
        cols=1,
        x_title="Test 1 Score",
    )
    fig.update_layout(title="Microchip QA Test 2 Score against Test 1 Score")

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
    fig.update_yaxes(title_text="Test 2 Score", row=1, col=1)

    fig.write_html("microchip_qa_visualization.html")


def sigmoid(z):
    """
    Logistic regression hypothesis is:
    h_theta(x) = g(theta^T x)
              1
    g(x) = -------
                -z
           1 + e
    """
    return 1.0 / (1.0 + math.exp(-1.0 * z))


def map_features(X1, X2):
    """
    Feature mapping function to polynomial features.

    `map_features(X1, X2)` maps the two input features to quadratic features.
    Returns a new feature array with more features, comprising of:
    1 (added to handle the intercept term)
    X1
    X2
    X1^2
    X2^2
    X1 * X2
    X1 * X2^2
    ...
    X1 * X2^5
    X2^6
    """
    if len(X1) != len(X2):
        assert "len(X1) != len(X2)"

    num_samples = len(X1)
    num_features = 28
    output = np.zeros(shape=(num_samples, num_features))
    for sample_idx in range(num_samples):
        xx1 = X1[sample_idx]
        xx2 = X2[sample_idx]
        result = map_feature_helper(xx1, xx2)
        print(result)
        output[sample_idx, :] = result[:]
    return output


def calc_hypothesis(theta, x):
    z = np.dot(theta, x)
    return sigmoid(z)


def calc_regularized_cost(theta, X, Y, lambda_param):
    (num_samples, num_features) = X.shape

    factor1 = 1.0 / num_samples
    sum1 = 0
    for sample_idx in range(num_samples):
        xx = X[sample_idx]
        yy = Y[sample_idx]
        h = calc_hypothesis(theta, xx)
        sum1 -= yy * math.log(h)
        sum1 -= (1 - yy) * math.log(1 - h)

    factor2 = float(lambda_param) / (2.0 * num_samples)
    sum2 = 0
    # do not regularize the theta[0] parameter
    for feature_idx in range(1, num_features):
        sum2 += math.pow(theta[feature_idx], 2)

    return factor1 * sum1 + factor2 * sum2


def calc_regularized_gradient_helper(theta, X, Y, feature_idx):
    (num_samples, num_features) = X.shape

    factor1 = 1.0 / num_samples
    sum = 0
    for sample_idx in range(num_samples):
        xx = X[sample_idx]
        yy = Y[sample_idx]
        h = calc_hypothesis(theta, xx)
        sum += (h - yy) * xx[feature_idx]
    return factor1 * sum


def calc_regularized_gradient(theta, X, Y, lambda_param):
    (num_samples, num_features) = X.shape
    gradient = np.zeros(num_features)
    gradient[0] = calc_regularized_gradient_helper(theta, X, Y, 0)
    for feature_idx in range(1, num_features):
        lambda_factor = float(lambda_param) / float(num_samples)
        gradient[feature_idx] = (
            calc_regularized_gradient_helper(theta, X, Y, feature_idx)
            + lambda_factor * theta[feature_idx]
        )
    return gradient


if __name__ == "__main__":
    [X, Y] = load_data("../assignment/ex2data2.txt")

    visualize_data(X, Y)

    mapped_X = map_features(X[:, 0], X[:, 1])
    (num_samples, num_features) = mapped_X.shape

    # Initialize fitting parameters
    initial_theta = np.zeros(num_features)

    initial_cost = calc_regularized_cost(initial_theta, mapped_X, Y, LAMBDA_PARAM)
    print("Cost at initial theta (zeros): ", initial_cost)
    initial_grad = calc_regularized_gradient(initial_theta, mapped_X, Y, LAMBDA_PARAM)
    print(
        "Gradient at initial theta (zeros) - first five values only:", initial_grad[0:5]
    )

    result = optimize.minimize(
        fun=calc_regularized_cost,
        x0=initial_theta,
        args=(mapped_X, Y, LAMBDA_PARAM),
        method="TNC",
        jac=calc_regularized_gradient,
    )
    optimal_theta = result.x
    print("Optimal theta: ", optimal_theta)
    visualize_data_with_decision_boundary(X, Y, optimal_theta)
