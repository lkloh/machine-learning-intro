#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

PIXELS_IN_HEIGHT = 20
PIXELS_IN_WIDTH = 20

NUM_IMAGE_ROWS = 10
NUM_IMAGE_COLS = 10


def display_image(all_images):
    pixels_for_display = np.ones(
        shape=(
            PIXELS_IN_HEIGHT * NUM_IMAGE_ROWS,
            PIXELS_IN_WIDTH * NUM_IMAGE_COLS,
        )
    )

    for image_row in range(NUM_IMAGE_ROWS):
        for image_col in range(NUM_IMAGE_COLS):
            image_index = image_row * NUM_IMAGE_ROWS + image_col
            raw_pixels = all_images[image_index, :]
            num_array = raw_pixels.reshape(PIXELS_IN_HEIGHT, PIXELS_IN_WIDTH)
            for pixel_row in range(PIXELS_IN_HEIGHT):
                for pixel_col in range(PIXELS_IN_WIDTH):
                    pixels_for_display[
                        image_row * PIXELS_IN_HEIGHT + pixel_row,
                        image_col * PIXELS_IN_WIDTH + pixel_col,
                    ] = num_array[pixel_row, pixel_col]

    display_colors = plt.imshow(
        pixels_for_display, cmap="gray", interpolation="nearest", origin="lower"
    )
    plt.colorbar(display_colors)

    plt.show()
