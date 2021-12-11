#!/usr/bin/env python3

import numpy as np
from scipy.io import loadmat
import display_image
from logistic_regression_cost_function import logistic_regression_cost_func

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

    # logistic regression
    lr_test_case_theta = np.array([-3, -1, 1, 2])
    lr_test_X = np.array([
        [1, 0.1, 0.6 , 1.1],
        [1, 0.2, 0.7, 1.2],
        [1, 0.3, 0.8 ,1.3],
        [1, 0.4, 0.9, 1.2],
        [1, 0.5, 1.0, 1.5],
    ])
    lr_test_Y = np.array([1, 0, 1, 0, 1])
    lr_test_lambda = 3
    lr_test_cost = logistic_regression_cost_func(lr_test_case_theta, lr_test_X, lr_test_Y, lr_test_lambda)
    print('expected cost: ', lr_test_cost)
