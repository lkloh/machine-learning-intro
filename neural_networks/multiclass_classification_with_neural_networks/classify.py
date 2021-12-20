#!/usr/bin/env python3

import numpy as np
from scipy.io import loadmat

HANDWRITTEN_DIGITS = loadmat("../assignment/ex3data1.mat")
WEIGHTS = loadmat("../assignment/ex3weights.mat")

NUM_LABELS = 10  # 10 labels, from 0, 2, ..., 9
INPUT_LAYER_SIZE = 400  # 20x20 Input Images of Digits
HIDDEN_LAYER_SIZE = 25

if __name__ == "__main__":
    X = np.array(HANDWRITTEN_DIGITS["X"])
    Y = np.array(HANDWRITTEN_DIGITS["y"])
    Y[Y == 10] = 0

    theta1 = WEIGHTS["Theta1"]
    theta2 = WEIGHTS["Theta2"]

    print(theta1)
