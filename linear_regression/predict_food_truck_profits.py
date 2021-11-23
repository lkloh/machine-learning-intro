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

def plot_data(independent_vars, dependent_vars):
    fig = make_subplots(
        rows=1,
        cols=1,
        x_title="Population",
    )
    fig.update_layout(title="Profit against Population")

    fig.append_trace(
        plotly_go.Scatter(
            x=independent_vars,
            y=dependent_vars,
            name="Profit against Population",
            mode="markers",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Profit in $10,000", row=1, col=1)

    fig.write_html("predict_food_truck_prices.html")


if __name__ == "__main__":
    [independent_vars, dependent_vars] = load_data("./instructions/ex1data1.txt")
    plot_data(independent_vars, dependent_vars)



