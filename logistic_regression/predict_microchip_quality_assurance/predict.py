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
            x=X_admitted[:, 1],
            y=X_admitted[:, 2],
            name="Admitted",
            mode="markers",
        ),
        row=1,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            x=X_rejected[:, 1],
            y=X_rejected[:, 2],
            name="Rejected",
            mode="markers",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Test 2 Score", row=1, col=1)

    fig.write_html("microchip_qa_visualization.html")



if __name__ == "__main__":
    [X, Y] = load_data("../assignment/ex2data2.txt")
    (num_samples, _) = X.shape
    # Add intercept term to X
    X = np.c_[np.ones(shape=(num_samples, 1)), X]

    visualize_data(X, Y)
