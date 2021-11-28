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


def visualize_data(X, Y):
    X_admitted = X[Y]

    Y_rejected = np.invert(Y)
    X_rejected = X[Y_rejected]

    fig = make_subplots(
        rows=1,
        cols=1,
        x_title="Test 1 Score",
    )
    fig.update_layout(title="Test 2 Score against Test 1 Score")

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

    degree = 6
    num_features = 28
    num_samples = len(X1)

    output = np.ones(shape=(num_samples, num_features))
    for sample_idx in range(num_samples):
        feature_index = 1
        xx1 = X1[sample_idx]
        xx2 = X2[sample_idx]
        for i in range(1, degree + 1):
            for j in range(0, i + 1):
                output[sample_idx, feature_index] = math.pow(xx1, i - j) * math.pow(
                    xx2, j
                )
                feature_index += 1
    return output


def cost_function(theta, X, Y, lambda_):
    pass


if __name__ == "__main__":
    [X, Y] = load_data("../assignment/ex2data2.txt")

    visualize_data(X, Y)

    mapped_X = map_features(X[:, 0], X[:, 1])

    print(mapped_X)
