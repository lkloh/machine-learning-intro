#!/usr/bin/env python3

import numpy as np
import math
from numpy.linalg import inv

def load_data(filename):
    file = open(filename, "r")
    contents = file.readlines()
    num_vars = len(contents)

    X = np.zeros(shape=(num_vars, 2))
    y = np.zeros(num_vars)
    for i, line in enumerate(contents):
        chunks = line.split(",")
        X[i][0] = float(chunks[0])
        X[i][1] = float(chunks[1])
        y[i] = float(chunks[2])
    return [X, y]

def calc_hypothesis(x_vars, theta):
    return theta[0] + theta[1] * x_vars[0] + theta[2] * x_vars[1]

if __name__ == "__main__":
    [X, y] = load_data("../instructions/ex1data2.txt")

    (num_samples, num_features) = X.shape

    # Add intercept term to X:
    modified_X = np.c_[np.ones(shape=(num_samples,1)), X]


    optimized_theta = inv(modified_X.transpose() @ modified_X) @ modified_X.transpose() @ y

    print("Estimated cost of a 1650 sq-ft, 3 br house:")
    estimated_price = calc_hypothesis([1650, 3], optimized_theta)
    print("$ %f" % round(estimated_price, 2))




