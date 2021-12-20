#!/usr/bin/env python3
import numpy as np
import math


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-1.0 * z))


def add_intercept_term(arr):
    (num_rows, _) = arr.shape
    return np.c_[np.ones(shape=(num_rows, 1)), arr]


def classification_accuracy(predicted, actual):
    if predicted.shape != actual.shape:
        return 0

    (num_samples, _) = predicted.shape
    is_same = predicted == actual
    print(predicted)
    accuracy = (is_same == True).sum() / float(num_samples)
    return accuracy * 100.0


def apply_hypothesis(arr):
    (num_samples, num_features) = arr.shape
    transformed_arr = np.zeros(shape=(num_samples, num_features))
    for sample_idx in range(num_samples):
        z = arr[sample_idx, :]
        transformed_arr[sample_idx, :] = list(map(sigmoid, z))
    return transformed_arr


def predict_classification(theta1, theta2, X):
    """
    Predicts the label of an input given the trained weights (theta1, theta2)
    of a neural network
    """

    (num_samples, _) = X.shape

    p = np.zeros(shape=(num_samples, 1))

    z2 = add_intercept_term(X) @ theta1.transpose()
    a2 = apply_hypothesis(z2)

    z3 = add_intercept_term(a2) @ theta2.transpose()
    a3 = apply_hypothesis(z3)

    for sample_idx in range(num_samples):
        hypotheses = a3[sample_idx, :]
        
        # theta1, theta2 were trained with indexing starting at 1
        p[sample_idx] = np.argmax(hypotheses, axis=0) + 1

    return p
