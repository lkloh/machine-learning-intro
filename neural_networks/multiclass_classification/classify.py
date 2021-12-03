#!/usr/bin/env python3

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
import scipy.optimize as optimize
from scipy.io import loadmat
import random
import display_image

HANDWRITTEN_DIGITS = loadmat("../assignment/ex3data1.mat")


if __name__ == "__main__":
    X = np.array(HANDWRITTEN_DIGITS["X"])
    raw_y = np.array(HANDWRITTEN_DIGITS["y"])
    Y = [10 if elem == 10 else elem for elem in raw_y]

    (num_samples, num_features) = X.shape

    # Randomly select 100 data points to display
    indices = [idx for idx in range(num_samples)]
    randomly_selected_indices = np.random.choice(
        indices,
        size=display_image.NUM_IMAGE_ROWS * display_image.NUM_IMAGE_COLS,
        replace=False,
    )
    randomly_selected_x = np.zeros(
        shape=(
            display_image.NUM_IMAGE_ROWS * display_image.NUM_IMAGE_COLS,
            num_features,
        )
    )
    for i in range(len(randomly_selected_indices)):
        randomly_selected_x[i, :] = X[randomly_selected_indices[i], :]

    display_image.show_image(randomly_selected_x)
