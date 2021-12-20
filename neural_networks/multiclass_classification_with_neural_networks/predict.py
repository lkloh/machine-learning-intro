#!/usr/bin/env python3
import numpy as np

def add_intercept_term(arr):
    (num_rows, _) = arr.shape

    new_arr = np.c_[np.ones(shape=(num_rows, 1)), arr]
    return new_arr

def predict_classification(theta1, theta2, X):
    '''
    Predicts the label of an input given the trained weights (theta1, theta2)
    of a neural network
    '''

    (num_samples, num_features) = X.shape
    (num_labels, _) = theta2.shape

    p = np.zeros(shape=(num_samples, 1))

    # Add intercept term to X
    X = add_intercept_term(X)

    hidden_layer = X @ theta1.transpose()
    hidden_layer = add_intercept_term(hidden_layer)

    output_layer = hidden_layer @ theta2.transpose()
    

    return p

