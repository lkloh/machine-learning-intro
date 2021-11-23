#!/usr/bin/env python3

import numpy as np
import random
from plotly.subplots import make_subplots
import plotly.graph_objects as plotly_go

def load_data(filename):
    file = open(filename, 'r')
    contents = file.readlines()
    num_vars = len(contents)

    independent_vars = [0] * num_vars
    dependent_vars = [0] * num_vars
    for i, line in enumerate(contents):
        chunks = line.split(',')
        independent_vars[i] = float(chunks[0])
        dependent_vars[i] = float(chunks[1])
    return [independent_vars, dependent_vars]

if __name__ == "__main__":
    [independent_vars, dependent_vars] = load_data("./instructions/ex1data1.txt")
    print(independent_vars)
