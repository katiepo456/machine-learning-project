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

    print(f'-------------------------------------------------------------------- OUTLIERS REMOVED FROM DATA -----------'
          f'---------------------------------------------------------')
    z_scores = stats.zscore(original_ys)
    new_xs: list[list[float]] = []
    new_ys: list[float] = []
    for x in range(len(z_scores)):
        if -2 < z_scores[x] < 2:
            new_xs.append(original_xs[x])
            new_ys.append(original_ys[x])
    reg = linear_model.LinearRegression(fit_intercept=True)
    xs = new_xs
    ys = new_ys
    reg.fit(xs, ys)

    print(f'y = {round(reg.intercept_, 2)} + {round(reg.coef_[0], 2)}(time_frame) + '
          f'{round(reg.coef_[1], 2)}(relatedness) + {round(reg.coef_[2], 2)}(studied_abroad) + '
          f'{round(reg.coef_[3], 2)}(student_leadership_position) + {round(reg.coef_[4], 2)}(center_of_experience) + '
          f'{round(reg.coef_[5], 2)}(number_of_internships) + {round(reg.coef_[6], 2)}(gpa)')
    print(f'r-squared: {reg.score(xs, ys)}')
