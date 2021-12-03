#!/usr/bin/env python3

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
import scipy.optimize as optimize
from scipy.io import loadmat
import random
from display_image import display_image

HANDWRITTEN_DIGITS = loadmat("../assignment/ex3data1.mat")


if __name__ == "__main__":
    X = np.array(HANDWRITTEN_DIGITS["X"])
    raw_y = np.array(HANDWRITTEN_DIGITS["y"])
    Y = [10 if elem == 10 else elem for elem in raw_y]

    (num_samples, num_features) = X.shape

    # Randomly select 100 data points to display
    indices = [idx for idx in range(num_samples)]
    randomly_selected_indices = np.random.choice(indices, size=100, replace=False)
    randomly_selected_x = X[indices, :]

    display_image(randomly_selected_x)
