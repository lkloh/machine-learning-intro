#!/usr/bin/env python3

import numpy as np
from scipy.io import loadmat
from predict import predict_classification, classification_accuracy

HANDWRITTEN_DIGITS = loadmat("../assignment/ex3data1.mat")
WEIGHTS = loadmat("../assignment/ex3weights.mat")

NUM_LABELS = 10  # 10 labels, from 0, 2, ..., 9
INPUT_LAYER_SIZE = 400  # 20x20 Input Images of Digits
HIDDEN_LAYER_SIZE = 25

if __name__ == "__main__":
    X = np.array(HANDWRITTEN_DIGITS["X"])
    Y = np.array(HANDWRITTEN_DIGITS["y"])

    # 25 units in 2nd layer, dimensions 25 x 401
    theta1 = WEIGHTS["Theta1"]

    # 10 output units for the 10 digit classes, size 10 x 26
    theta2 = WEIGHTS["Theta2"]

    p = predict_classification(theta1, theta2, X)
    print("Training Set Accuracy percentage:  ", classification_accuracy(p, Y))
