#!/usr/bin/env python3

import numpy as np
from scipy.io import loadmat

HANDWRITTEN_DIGITS = loadmat("../assignment/ex3data1.mat")

NUM_LABELS = 10  # 10 labels, from 1, 2, ..., 9, 10

if __name__ == "__main__":
    X = np.array(HANDWRITTEN_DIGITS["X"])
    Y = np.array(HANDWRITTEN_DIGITS["y"])



