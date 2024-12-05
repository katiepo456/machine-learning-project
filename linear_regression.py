# linear_regression.py
# Machine Learning Group 3 @Champlain College

import csv
import statistics
from scipy import stats
from pathlib import Path
from sklearn import linear_model
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np


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


# Plot a 2D scatter plot with a line of best fit
#   Generated by claude.ai
def plot_2d(xs: list[float], ys: list[float], title: str) -> None:
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    reg_line = slope * x + intercept

    plt.scatter(x, y)
    plt.plot(x, reg_line, color="red")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)

    plt.show()


# Plot a 3D scatter plot with a plane of best fit
#   Generated by claude.ai
def plot_3d(xs1: list[float], xs2: list[float], ys: list[float], title: str) -> None:
    # Convert to numpy arrays
    x1 = np.array(xs1, dtype=float)
    x2 = np.array(xs2, dtype=float)
    y = np.array(ys, dtype=float)

    # Create the 3D figure and axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(x1, x2, y, c='blue', marker='o')

    # Fit a plane (linear regression in 3D)
    # Create a mesh grid for the plane
    A = np.c_[x1, x2, np.ones(len(x1))]
    plane_coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    # Create a mesh grid for the plane visualization
    xx, yy = np.meshgrid(np.linspace(x1.min(), x1.max(), 30),
                         np.linspace(x2.min(), x2.max(), 30))
    z_plane = plane_coeffs[0] * xx + plane_coeffs[1] * yy + plane_coeffs[2]

    # Plot the regression plane
    ax.plot_surface(xx, yy, z_plane, alpha=0.5, color="red")

    # Labels and title
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Y")
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


def run_regression_model(original_xs: list[list[float]], original_ys: list[float]) -> tuple[list[float], float, float]:
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


def remove_outliers(original_xs: list[list[float]], original_ys: list[float]) -> tuple[list[list[float]], list[float]]:
    z_scores = stats.zscore(original_ys)
    cleaned_xs: list[list[float]] = []
    cleaned_ys: list[float] = []
    for x in range(len(z_scores)):
        if -3 < z_scores[x] < 3:
            cleaned_xs.append(original_xs[x])
            cleaned_ys.append(original_ys[x])
    return cleaned_xs, cleaned_ys


def exclude_parameters_from_model(path: Path, exceptions: list[float]) -> float:
    xs, ys = read_file(path, exceptions)
    cleaned_xs, cleaned_ys = remove_outliers(xs, ys)
    coefficients, y_intercept, r_squared = run_regression_model(cleaned_xs, cleaned_ys)
    return r_squared


def run_regression_model_on_major(file_path: Path) -> None:
    original_xs, original_ys = read_file(file_path, [])

    print(f'---------- ALL DATA POINTS ----------')
    coefficients, y_intercept, r_squared = run_regression_model(original_xs, original_ys)
    display_regression_equation(coefficients, y_intercept, r_squared)

    print(f'\n---------- OUTLIERS REMOVED FROM DATA ----------')
    cleaned_xs, cleaned_ys = remove_outliers(original_xs, original_ys)
    coefficients2, y_intercept2, r_squared2 = run_regression_model(cleaned_xs, cleaned_ys)
    display_regression_equation(coefficients2, y_intercept2, r_squared2)

    print(f'\n---------- REGRESSION MODEL FOR GPA & INTERNSHIPS ----------')
    gpa = []
    number_of_internships = []
    for row in cleaned_xs:
        gpa.append(row[6])
        number_of_internships.append(row[5])

    xs, ys = read_file(file_path, [1, 2, 3, 4, 5])
    cleaned_xs, cleaned_ys = remove_outliers(xs, ys)
    coefficients, y_intercept, r_squared = run_regression_model(cleaned_xs, cleaned_ys)
    print(f'y = {y_intercept} + '
          f'{coefficients[0]}(gpa) + '
          f'{coefficients[1]}(number_of_internships)')
    print(f'r-squared: {r_squared}')

    # plot_3d(gpa, number_of_internships, cleaned_ys, "Impact of GPA & Internships on Starting Salary")

    print(f'\n---------- EXCLUDING PARTICULAR PARAMETERS ----------')
    for i in range(0, 7):
        parameters_to_exclude = list(combinations([1, 2, 3, 4, 5, 6, 7], i))
        for parameter_list in parameters_to_exclude:
            parameter_map = {
                1: "(Time Frame X)",
                2: "(Relatedness X)",
                3: "(Studied Abroad X)",
                4: "(Student Leadership X)",
                5: "(Center of Experience X)",
                6: "(Internships X)",
                7: "(GPA X)"
            }
            excluded_parameters = [parameter_map[param] for param in parameter_list if param in parameter_map]
            print(f'{exclude_parameters_from_model(csi_path, list(parameter_list))} '
                  f'{", ".join(excluded_parameters)}')
        print("-----------------------------------------------------")


if __name__ == '__main__':
    print(f'\n--------------------------------- COMPUTER SCIENCE & INNOVATION ALUMNI ---------------------------------')
    csi_path = Path(__file__).with_name('csi_alum_data.csv')
    run_regression_model_on_major(csi_path.absolute())

    print(f'\n------------------------------ COMPUTER NETWORKING & CYBERSECURITY ALUMNI ------------------------------')
    csi_path = Path(__file__).with_name('cncs_alum_data.csv')
    run_regression_model_on_major(csi_path.absolute())

    print(f'\n--------------------------------- COMPUTER & DIGITAL FORENSICS ALUMNI ----------------------------------')
    csi_path = Path(__file__).with_name('cdf_alum_data.csv')
    run_regression_model_on_major(csi_path.absolute())
