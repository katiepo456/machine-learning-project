# linear_regression.py
# Machine Learning Group 3 @Champlain College

import csv
import copy
import statistics
import numpy as np
from typing import Callable
from math import ceil
from pathlib import Path
from sklearn import linear_model


# Read a CSV file
# The first column is the salary (our y values)
# The second column is the indicated time frame for securing the position
# The rest of the columns are the remaining x parameters (relatedness_to_career_goals, study_abroad?,
#   student_leadership_position?, center_of_experience?, number_of_internships, GPA)
# Note that xs here will be a list of lists because each data point has multiple x values
#   (that's why it's called a multivariate linear regression)
# Based on starter code in ps6.py
def read_file(file_path: Path) -> tuple[list[list[float]], list[float]]:
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        xs = []
        ys = []
        for row in reader:
            x_list = []
            ys.append(float(row[0]))      # salary
            x_list.append(float(row[1]))  # time frame for securing position
            x_list.append(float(row[2]))  # relatedness to career goals
            x_list.append(float(row[3]))  # studied abroad?
            x_list.append(float(row[4]))  # held a student leadership position?
            x_list.append(float(row[5]))  # worked with center of experience?
            x_list.append(float(row[6]))  # number of internships
            x_list.append(float(row[7]))  # GPA
            xs.append(x_list)

        return xs, ys


# Find the mean of a list of numbers
# Created by GitHub Copilot
def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


# Find the covariance of two lists of numbers
# Created with assistance from GitHub Copilot
def covariance(xs: list[float], ys: list[float]) -> float:
    x_mean = mean(xs)
    y_mean = mean(ys)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / len(xs)


# Find the variance of a list of numbers
# Created with assistance from GitHub Copilot
def variance(xs: list[float]) -> float:
    x_mean = mean(xs)
    return sum((x - x_mean) ** 2 for x in xs) / len(xs)


# Find the standard deviation of a list of numbers
# Created with assistance from GitHub Copilot
def standard_deviation(xs: list[float]) -> float:
    return variance(xs) ** 0.5


# Find the y-value of a line of best fit at a given set of x-values
# Created with assistance from GitHub Copilot
def predict(x: list[float], ys: list[float], theta: list[float]) -> float:
    return theta[0] + sum(theta[i + 1] * x[i] for i in range(len(x)))


# Find the r-squared value of a line of best fit
# Created with assistance from GitHub Copilot
def r_squared(xs: list[list[float]], ys: list[float], theta: list[float]) -> float:
    y_mean = mean(ys)
    return 1 - sum((y - predict(x, ys, theta)) ** 2 for x, y in zip(xs, ys)) / sum((y - y_mean) ** 2 for y in ys)


# Normalize the values by subtracting the mean and dividing by the standard deviation
# Created with assistance from GitHub Copilot
def normalize_xs(xs: list[list[float]]) -> list[list[float]]:
    m = len(xs[0])
    normalized_xs = []
    x_stds = []
    x_means = []
    for i in range(m):
        x = [x[i] for x in xs]
        x_means.append(mean(x))
        x_stds.append(standard_deviation(x))
    for x in xs:
        normalized_xs.append([(x[i] - x_means[i]) / x_stds[i] for i in range(m)])
    return normalized_xs


# Normalize the values by subtracting the mean and dividing by the standard deviation
def normalize_ys(ys: list[float]) -> list[float]:
    y_mean = mean(ys)
    y_std = standard_deviation(ys)
    return [(y - y_mean) / y_std for y in ys]


# Find the cost of a line of best fit for the multivariate linear regression
# Created with assistance from GitHub Copilot
def cost(xs: list[list[float]], ys: list[float], theta: list[float]) -> float:
    m = len(xs[0])
    n = len(xs)
    return sum((theta[0] + sum(theta[i + 1] * x[i] for i in range(m)) - y) ** 2 for x, y in zip(xs, ys)) / 2 * n


# Plot the costs of a line of best fit for the multivariate linear regression
# from gradient descent over time
# Created with assistance from GitHub Copilot
# Requires matplotlib (pip or pip3 install matplotlib)
def plot_costs(costs: list[float]) -> None:
    import matplotlib.pyplot as plt
    plt.plot(range(len(costs)), costs)
    plt.show()


# Find the theta values for a line of best fit for the multivariate linear regression
# Returns theta values and costs over time
# Created with assistance from GitHub Copilot
def gradient_descent(xs: list[list[float]], ys: list[float], theta: list[float], alpha: float = 0.000004,
                     iterations: int = 100000, convergence: float = 0.1) -> tuple[list[float], list[float]]:
    m = len(xs[0])
    n = len(xs)
    costs = []
    for iteration in range(iterations):
        new_theta = [theta[0] + alpha * sum(y - predict(x, ys, theta) for x, y in zip(xs, ys))]
        for i in range(m):
            new_theta.append(theta[i + 1] + alpha * sum((y - (predict(x, ys, theta))) * x[i] for x, y in zip(xs, ys)))
        theta = new_theta
        costs.append(cost(xs, ys, theta))
        if iteration > 1 and abs(costs[-1] - costs[-2]) < convergence:
            print(f"Converged after {iteration} iterations")
            break

    return theta, costs


def starter_run():
    p = Path(__file__).with_name('csi_alum_data.csv')
    original_xs, original_ys = read_file(p.absolute())


    normalized_xs = normalize_xs(original_xs)
    normalized_ys = normalize_ys(original_ys)
    initial_theta = [0.0] * (len(normalized_xs[0]) + 1)
    normalized_theta, normalized_costs = gradient_descent(normalized_xs, normalized_ys, initial_theta)
    """

    xs = original_xs
    ys = original_ys
    initial_theta = [0.0] * (len(xs[0]) + 1)
    theta, costs = gradient_descent(xs, ys, initial_theta)
    """

    print(f'--------------------------------------------------------------------------------')
    print(f'y = {normalized_theta[0]} + {normalized_theta[1]}studied_abroad + '
          f'{normalized_theta[2]}student_leadership + {normalized_theta[3]}internships + {normalized_theta[4]}gpa')
    print(f'r-squared: {r_squared(normalized_xs, normalized_ys, normalized_theta)}')

    plot_costs(normalized_costs)


if __name__ == '__main__':
    p = Path(__file__).with_name('csi_alum_data.csv')
    original_xs, original_ys = read_file(p.absolute())

    reg = linear_model.LinearRegression(fit_intercept=True)
    xs = original_xs
    ys = original_ys
    reg.fit(xs, ys)

    print(f'------------------------------------------------------------------------- ALL DATA POINTS ----------------'
          f'---------------------------------------------------------')
    print(f'y = {round(reg.intercept_,2)} + {round(reg.coef_[0],2)}(time_frame) + '
          f'{round(reg.coef_[1],2)}(relatedness) + {round(reg.coef_[2],2)}(studied_abroad) + '
          f'{round(reg.coef_[3],2)}(student_leadership_position) + {round(reg.coef_[4],2)}(center_of_experience) + '
          f'{round(reg.coef_[5],2)}(number_of_internships) + {round(reg.coef_[6],2)}(gpa)')
    print(f'r-squared: {reg.score(xs, ys)}')

    print(f'Outliers Removed from Data')
    std_dev = np.std(original_xs, axis=0)

    print(std_dev)
