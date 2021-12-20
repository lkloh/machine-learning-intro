#!/usr/bin/env python3
import numpy as np


def predict_classification(theta1, theta2, X):
    '''
    Predicts the label of an input given the trained weights (theta1, theta2)
    of a neural network
    '''

    (num_samples, num_features) = X.shape
    (num_labels, _) = theta2.shape

    p = np.zeros(shape=(num_samples, 1))

    return p

