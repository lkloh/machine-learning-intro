#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

PIXELS_IN_HEIGHT = 20
PIXELS_IN_WIDTH = 20


def display_image(X):
    row = X[1, :]

    num_array = np.reshape(row, (PIXELS_IN_HEIGHT, PIXELS_IN_WIDTH))
    print(num_array)

    c = plt.imshow(num_array, cmap="gray", interpolation="nearest", origin="lower")
    plt.colorbar(c)

    plt.show()
