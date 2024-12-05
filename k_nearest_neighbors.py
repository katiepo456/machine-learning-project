import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def get_info(X_train, X_test, y_train, y_test):
    for i, weights in enumerate(["uniform", "distance"]):
        knn_regressor = KNeighborsRegressor(n_neighbors=5, weights=weights)

        knn_regressor.fit(X_train, y_train)

        y_pred = knn_regressor.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse} for weights = {weights}")

        r2 = r2_score(y_test, y_pred)
        print(f"R-squared: {r2} for weights = {weights}")

        rmse = np.sqrt(mse)
        print(f"Root Mean Squared Error: {rmse} for weights = {weights}")


        plt.subplot(2, 2, (i*2) + 1)
        plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Ideal fit')
        plt.xlabel('Actual Salary')
        plt.ylabel('Predicted Salary')
        plt.legend()
        plt.title("KNN Regression: Predicted vs Actual: weights = '%s')" % (weights))

        plt.subplot(2, 2, (i*2) + 2)
        plt.scatter(X_train.GPA, y_train, color="darkorange", label="data")
        plt.plot(X_test.GPA, y_pred, color="navy", label="prediction")
        plt.axis("tight")
        plt.xlabel('GPA')
        plt.ylabel('Salary')
        plt.legend()
        plt.title("KNeighborsRegressor (k = 5, weights = '%s')" % (weights))


        param_grid = {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # Distance metric: 1 for Manhattan, 2 for Euclidean
        }

        # GridSearchCV
        grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        # Best parameters
        print(f"Best parameters: {grid_search.best_params_}")

        # Best model
        best_knn = grid_search.best_estimator_

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv('csi_alum_data.csv')

    # X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X = data.drop('salary', axis=1)  # Drop the 'salary' column to get the feature set
    y = data['salary']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    get_info(X_train, X_test, y_train, y_test)
    #make_graph(X_train, X_test, y)
