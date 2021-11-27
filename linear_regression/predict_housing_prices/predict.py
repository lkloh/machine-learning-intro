#!/usr/bin/env python3

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

ALPHA = 0.01
NUM_ITERATIONS = 1500


def load_data(filename):
    file = open(filename, "r")
    contents = file.readlines()
    num_vars = len(contents)

    X = np.zeros(shape=(num_vars, 2))
    y = np.zeros(shape=(num_vars, 1))
    for i, line in enumerate(contents):
        chunks = line.split(",")
        X[i][0] = float(chunks[0])
        X[i][1] = float(chunks[1])
        y[i][0] = float(chunks[2])
    return [X, y]


if __name__ == "__main__":
    [X, y] = load_data("../instructions/ex1data2.txt")

    print(X)
