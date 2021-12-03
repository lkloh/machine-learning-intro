#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

PIXELS_IN_HEIGHT = 20
PIXELS_IN_WIDTH = 20

NUM_ROWS = 10
NUM_COLS = 10


def display_image(all_images):
    pixels_for_display = np.ones(
        shape=(
            PIXELS_IN_HEIGHT * NUM_ROWS,
            PIXELS_IN_WIDTH * NUM_COLS,
        )
    )

    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            image_index = row * NUM_ROWS + col
            raw_pixels = all_images[image_index, :]
            num_array = np.reshape(raw_pixels, (PIXELS_IN_HEIGHT, PIXELS_IN_WIDTH))
            for row_idx in range(PIXELS_IN_HEIGHT):
                for col_idx in range(PIXELS_IN_WIDTH):
                    pixels_for_display[row * PIXELS_IN_WIDTH + row_idx, col * PIXELS_IN_HEIGHT + col_idx] = num_array[row_idx, col_idx]
    

    display_colors = plt.imshow(
        pixels_for_display, cmap="gray", interpolation="nearest", origin="lower"
    )
    plt.colorbar(display_colors)

    plt.show()
