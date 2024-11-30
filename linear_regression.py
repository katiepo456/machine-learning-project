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


# Read a CSV file
# The first column is the salary (our y values)
# The second column is the indicated time frame for securing the position
# The rest of the columns are the remaining x parameters (relatedness_to_career_goals, study_abroad?,
#   student_leadership_position?, center_of_experience?, number_of_internships, GPA)
# Note that xs here will be a list of lists because each data point has multiple x values
#   (that's why it's called a multivariate linear regression)
# Based on starter code in ps6.py
def read_complete_file(file_path: Path) -> tuple[list[list[float]], list[float]]:
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


def read_file_except(file_path: Path, exceptions: list[float]) -> tuple[list[list[float]], list[float]]:
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        xs = []
        ys = []
        for row in reader:
            x_list = []
            ys.append(float(row[0]))      # salary
            if 1 not in exceptions:
                x_list.append(float(row[1]))  # time frame for securing position
            if 2 not in exceptions:
                x_list.append(float(row[2]))  # relatedness to career goals
            if 3 not in exceptions:
                x_list.append(float(row[3]))  # studied abroad?
            if 4 not in exceptions:
                x_list.append(float(row[4]))  # held a student leadership position?
            if 5 not in exceptions:
                x_list.append(float(row[5]))  # worked with center of experience?
            if 6 not in exceptions:
                x_list.append(float(row[6]))  # number of internships
            if 7 not in exceptions:
                x_list.append(float(row[7]))  # GPA
            xs.append(x_list)

        return xs, ys


def run_regression_model(original_xs: list[list[float]], original_ys:list[float]) -> tuple[list[float], float, float]:
    regression = linear_model.LinearRegression(fit_intercept=True)
    xs = original_xs
    ys = original_ys
    regression.fit(xs, ys)

    coefficients = [round(v, 2) for v in regression.coef_]
    y_intercept = round(regression.intercept_, 2)
    r_squared = regression.score(xs, ys)

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
    new_xs: list[list[float]] = []
    new_ys: list[float] = []
    for x in range(len(z_scores)):
        if -3 < z_scores[x] < 3:
            new_xs.append(original_xs[x])
            new_ys.append(original_ys[x])
    return new_xs, new_ys


if __name__ == '__main__':
    p = Path(__file__).with_name('csi_alum_data.csv')
    original_xs, original_ys = read_complete_file(p.absolute())

    print(f'------------------------------------------------------------------------- ALL DATA POINTS ----------------'
          f'---------------------------------------------------------')
    coefficients, y_intercept, r_squared = run_regression_model(original_xs, original_ys)
    display_regression_equation(coefficients, y_intercept, r_squared)

    print(f'-------------------------------------------------------------------- OUTLIERS REMOVED FROM DATA -----------'
          f'---------------------------------------------------------')
    cleaned_xs, cleaned_ys = remove_outliers(original_xs, original_ys)
    coefficients2, y_intercept2, r_squared2 = run_regression_model(cleaned_xs, cleaned_ys)
    display_regression_equation(coefficients2, y_intercept2, r_squared2)



    path = Path(__file__).with_name('csi_alum_data.csv')
    this_xs, this_ys = read_file_except(path.absolute(), [])
    cleaned_xs, cleaned_ys = remove_outliers(this_xs, this_ys)
    coefficients2, y_intercept2, r_squared2 = run_regression_model(cleaned_xs, cleaned_ys)

    print(f'r-squared: {r_squared2}')
