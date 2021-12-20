#!/usr/bin/env python3

import numpy as np
from logistic_regression_cost_function import sigmoid


def compute_prediction_accuracy(predictions, actual):
    if predictions.shape != actual.shape:
        return 0

    (num_samples, _) = predictions.shape
    num_success = 0
    for sample_idx in range(num_samples):
        predicted_label = (
            10 if predictions[sample_idx] == 0 else predictions[sample_idx]
        )
        if predicted_label == actual[sample_idx]:
            num_success += 1
    return float(num_success) / float(num_samples) * 100


def one_vs_all_predictions(all_theta, X):
    """
    Predict the label for a trained one-vs-all classified.

    The labels are in the range 1, 2, ..., K
    where K = len(all_theta)

    This function returns a vector of predictions for each example in the matrix X.
    """
    (num_samples, num_features) = X.shape
    (num_labels, _) = all_theta.shape

    # Add intercept term to X
    X = np.c_[np.ones(shape=(num_samples, 1)), X]

    predictions = np.zeros(shape=(num_samples, 1))

    Z = X @ all_theta.transpose()
    for sample_idx in range(num_samples):
        sample_z = Z[sample_idx, :]
        all_hypothesis = list(map(sigmoid, sample_z))
        predictions[sample_idx] = all_hypothesis.index(max(all_hypothesis))
    return predictions
