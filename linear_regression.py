# linear_regression.py
# Machine Learning Group 3 @Champlain College

import csv
import copy
import statistics
import numpy as np
from scipy import stats
from typing import Callable
from math import ceil
from pathlib import Path
from sklearn import linear_model


# Read a CSV file (the file is structured as follows):
#   0 - salary
#   1 - time frame for securing position
#   2 - relatedness to career goals
#   3 - whether or not they studied abroad
#   4 - whether or not they held a student leadership position
#   5 - if they worked at a center of experience
#   6 - number of internships
#   7 - GPA
# The first column is our y values (the salary) and the remaining columns are our x parameters
#   Note that xs here will be a list of lists because each data point has multiple x values
def read_file(file_path: Path, exceptions: list[float]) -> tuple[list[list[float]], list[float]]:
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        xs, ys = [], []
        for row in reader:
            ys.append(float(row[0]))  # salary

            # read through file and exclude indicated columns
            x_list = [float(row[i]) for i in range(1, 8) if i not in exceptions]
            xs.append(x_list)

        return xs, ys


def run_regression_model(original_xs: list[list[float]], original_ys:list[float]) -> tuple[list[float], float, float]:
    regression = linear_model.LinearRegression(fit_intercept=True)
    xs = original_xs
    ys = original_ys
    regression.fit(xs, ys)

    coefficients = [round(v, 2) for v in regression.coef_]
    y_intercept = round(regression.intercept_, 2)
    r_squared = round(regression.score(xs, ys), 4)

    return coefficients, y_intercept, r_squared


def display_regression_equation(coefficients: list[float], y_intercept: float, r_squared: float) -> None:
    print(f'y = {y_intercept} + '
          f'{coefficients[0]}(time_frame) + '
          f'{coefficients[1]}(relatedness) + '
          f'{coefficients[2]}(studied_abroad) + '
          f'{coefficients[3]}(student_leadership_position) + '
          f'{coefficients[4]}(center_of_experience) + '
          f'{coefficients[5]}(number_of_internships) + '
          f'{coefficients[6]}(gpa)')
    print(f'r-squared: {r_squared}')


def remove_outliers(original_xs: list[list[float]], original_ys:list[float]) -> tuple[list[list[float]], list[float]]:
    z_scores = stats.zscore(original_ys)
    cleaned_xs: list[list[float]] = []
    cleaned_ys: list[float] = []
    for x in range(len(z_scores)):
        if -3 < z_scores[x] < 3:
            cleaned_xs.append(original_xs[x])
            cleaned_ys.append(original_ys[x])
    return cleaned_xs, cleaned_ys


if __name__ == '__main__':
    p = Path(__file__).with_name('csi_alum_data.csv')
    original_xs, original_ys = read_file(p.absolute(), [])

    print(f'\n---------- ALL DATA POINTS ----------')
    coefficients, y_intercept, r_squared = run_regression_model(original_xs, original_ys)
    display_regression_equation(coefficients, y_intercept, r_squared)

    print(f'\n---------- OUTLIERS REMOVED FROM DATA ----------')
    cleaned_xs, cleaned_ys = remove_outliers(original_xs, original_ys)
    coefficients2, y_intercept2, r_squared2 = run_regression_model(cleaned_xs, cleaned_ys)
    display_regression_equation(coefficients2, y_intercept2, r_squared2)

    print(f'\n---------- EXCLUDING PARTICULAR PARAMETERS ----------')
    path = Path(__file__).with_name('csi_alum_data.csv')
    this_xs, this_ys = read_file(path.absolute(), [])
    cleaned_xs, cleaned_ys = remove_outliers(this_xs, this_ys)
    coefficients2, y_intercept2, r_squared2 = run_regression_model(cleaned_xs, cleaned_ys)

    print(f'r-squared: {r_squared2}')
