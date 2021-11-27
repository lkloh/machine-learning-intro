#!/usr/bin/env python3

import numpy as np
import random
from plotly.subplots import make_subplots
import plotly.graph_objects as plotly_go
import compute_cost
import gradient_descent

ALPHA = 0.01
NUM_ITERATIONS = 1500


def load_data(filename):
    file = open(filename, "r")
    contents = file.readlines()
    num_vars = len(contents)

    independent_vars = [0] * num_vars
    dependent_vars = [0] * num_vars
    for i, line in enumerate(contents):
        chunks = line.split(",")
        independent_vars[i] = float(chunks[0])
        dependent_vars[i] = float(chunks[1])
    return [independent_vars, dependent_vars]


def plot_data(independent_vars, dependent_vars, fig_title, theta=None):
    # num training examples
    m = len(dependent_vars)

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

    if theta is not None:
        x_min = round(min(independent_vars))
        x_max = round(max(dependent_vars))

        fitted_x = [xi for xi in range(x_min, x_max)]
        fitted_y = [(theta[0][0] + theta[1][0] * xi) for xi in range(x_min, x_max)]

        fig.append_trace(
            plotly_go.Scatter(
                x=fitted_x,
                y=fitted_y,
                name="Best-fit",
                mode="lines",
            ),
            row=1,
            col=1,
        )

    fig.write_html(fig_title + ".html")


def plot_cost_function(iterations, cost_function):
    fig = make_subplots(
        rows=1,
        cols=1,
        x_title="Population",
    )
    fig.update_layout(title="Cost function against number of iterations")

    fig.append_trace(
        plotly_go.Scatter(
            x=iterations,
            y=cost_function,
            name="Cost function against number of iterations",
            mode="markers",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Cost function", row=1, col=1)

    fig.write_html("cost_function_against_number_of_iterations.html")


def visualize_cost_function_against_params(X, y):
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    J_vals = np.zeros(shape=(len(theta0_vals), len(theta1_vals)))
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            theta = [theta0_vals[i], theta1_vals[j]]
            J_vals[i, j] = compute_cost.calc_cost(X, y, theta)

    # Plot 3D surface plot
    fig = plotly_go.Figure(data=[plotly_go.Surface(z=J_vals)])
    # fig.update_yaxes(title_text="Cost function", row=1, col=1)
    fig.update_layout(title="Cost function against parameters")
    fig.write_html("cost_function_against_parameters.html")


if __name__ == "__main__":
    [X, y] = load_data("../instructions/ex1data1.txt")

    plot_data(X, y, "price_against_population")

    theta = np.zeros(shape=(2, 1))
    J = compute_cost.calc_cost(X, y, theta)
    print("Cost function when theta is [%f, %f] is %f" % (theta[0], theta[1], J))

    theta_test = np.zeros(shape=(2, 1))
    theta_test[0] = -1
    theta_test[1] = 2
    J = compute_cost.calc_cost(X, y, [-1, 2])
    print(
        "Cost function when theta is [%f, %f] is %f" % (theta_test[0], theta_test[1], J)
    )

    [
        cost_function_history,
        gradient_descent_theta,
    ] = gradient_descent.calc_gradient_descent(X, y, theta, ALPHA, NUM_ITERATIONS)
    iterations = [i for i in range(1, NUM_ITERATIONS + 1)]
    plot_cost_function(iterations, cost_function_history)
    print(
        "Theta found by gradient descent: [%f, %f]"
        % (gradient_descent_theta[0], gradient_descent_theta[1])
    )

    plot_data(
        X, y, "price_against_population_linear_regression", gradient_descent_theta
    )

    visualize_cost_function_against_params(X, y)
