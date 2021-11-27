#!/usr/bin/env python3
import numpy as np
import math


def calc_hypothesis(x, theta):
    return theta[0] + theta[1] * x


"""
Computes the cost of using theta as the parameter for linear regression 
to fit the data points in X and y
"""


def calc_cost(X, y, theta):
    # number of training examples
    m = len(y)

    factor = 1.0 / (2.0 * m)

    sum_squared = 0
    for i in range(m):
        xi = X[i]
        yi = y[i]
        sum_squared += math.pow((calc_hypothesis(xi, theta) - yi), 2)

    return factor * sum_squared
