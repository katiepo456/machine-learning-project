# Machine Learning Group 3

from linear_regression import linear_regression
from random_forest_regressor import random_forest_model

if __name__ == '__main__':
    print(f'\n\n---------------- K NEAREST NEIGHBORS ----------------')
    # run KNN
    print(f'-----------------------------------------------------')

    print(f'\n\n------------------- RANDOM FOREST -------------------')
    random_forest_model()
    print(f'-----------------------------------------------------')

    print(f'\n\n-------------- LINEAR REGRESSION MODEL --------------')
    linear_regression()
