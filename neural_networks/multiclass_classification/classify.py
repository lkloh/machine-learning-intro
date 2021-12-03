#!/usr/bin/env python3

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
import scipy.optimize as optimize
from scipy.io import loadmat

HANDWRITTEN_DIGITS = loadmat("../assignment/ex3data1.mat")



if __name__ == "__main__":
    X = np.array(HANDWRITTEN_DIGITS['X'])
    Y = np.array(HANDWRITTEN_DIGITS['y'])
    print(X)

