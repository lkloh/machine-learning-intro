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
    y = np.zeros(num_vars)
    for i, line in enumerate(contents):
        chunks = line.split(",")
        X[i][0] = float(chunks[0])
        X[i][1] = float(chunks[1])
        y[i] = float(chunks[2])
    return [X, y]

def plot_price_vs_feature(x, y, feature):
    fig = make_subplots(
        rows=1,
        cols=1,
        x_title=feature,
    )
    fig.update_layout(title="Price of house vs %s" % feature)

    fig.append_trace(
        go.Scatter(
            x=x,
            y=y,
            name=feature,
            mode="markers",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Price of house", row=1, col=1)

    fig.write_html("price_of_house_vs_" + feature + ".html")

'''
Returns a normalized version of X where the mean value of each feature is 0,
and standard deviation is 1.
'''
def normalize_features(X):
    (n_samples, n_features) = X.shape
    normalized_X = np.zeros(shape=(n_samples, n_features))
    mu = np.zeros(n_features)
    sigma = np.zeros(n_features)

    for feature_idx in range(n_features):
        samples = X[:, feature_idx]

        mu[feature_idx] = np.mean(samples)
        sigma[feature_idx] = np.std(samples)

        for sample_idx in range(n_samples):
            normalized_X[sample_idx, feature_idx] = (X[sample_idx, feature_idx] - mu[feature_idx]) / sigma[feature_idx]

    return [normalized_X, mu, sigma]



if __name__ == "__main__":
    [X, y] = load_data("../instructions/ex1data2.txt")

    plot_price_vs_feature(X[:,0], y, "square_footage")
    plot_price_vs_feature(X[:,1], y, "num_of_rooms")

    [normalized_X, mu, sigma] = normalize_features(X)

    print(normalized_X)