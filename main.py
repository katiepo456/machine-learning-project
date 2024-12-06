# Machine Learning Group 3

from linear_regression import linear_regression
from random_forest_regressor import random_forest
from k_nearest_neighbors import k_nearest_neighbors

if __name__ == '__main__':
    print(f'\n\n---------------- K NEAREST NEIGHBORS ----------------')
    k_nearest_neighbors()
    print(f'-----------------------------------------------------')

    print(f'\n\n------------------- RANDOM FOREST -------------------')
    random_forest()
    print(f'-----------------------------------------------------')

    print(f'\n\n-------------- LINEAR REGRESSION MODEL --------------')
    linear_regression()
