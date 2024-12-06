from typing import Dict, Any
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

def get_info(major, neighbors, X_train, X_test, y_train, y_test):

    for neighbor in neighbors:
        print(f"For {neighbor} Nieghbors")
        for i, weights in enumerate(["uniform", "distance"]):
            plt.figure(figsize=(16, 10))
            plt.suptitle(major)
            knn_regressor = KNeighborsRegressor(n_neighbors=neighbor, weights=weights)

            knn_regressor.fit(X_train, y_train)

            y_pred = knn_regressor.predict(X_test)

            r2 = round(r2_score(y_test, y_pred), 2)
            print(f"R-squared: {r2}, for weights = {weights}")

            """accuracy = round(knn_regressor.score(X_test, y_test), 2)
            print(f"Accuracy: {accuracy}, for weights = {weights}")"""

            mse = round(mean_squared_error(y_test, y_pred),2)
            print(f"Mean Squared Error: {mse}, for weights = {weights}")

            """rmse = round(np.sqrt(mse),2)
            print(f"Sqaure Root Mean Squared Error: {rmse}, for weights = {weights}")
    
            mae = round(mean_absolute_error(y_test, y_pred), 2)
            print(f"Mean Absolute Error: {mae}, for weights = {weights}")"""

            plt.subplot(2, 2, 1)
            plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Ideal fit')
            plt.axis("tight")
            plt.xlabel('Actual Salary')
            plt.ylabel('Predicted Salary')
            plt.legend()
            plt.title("KNN Regression: Predicted vs Actual: (k = %i, weights = '%s')" % (neighbor, weights))

            plt.subplot(2, 2, 2)
            plt.scatter(X_train.GPA, y_train, color="darkorange", label="data")
            plt.plot(X_test.GPA, y_pred, color="navy", label="prediction")
            plt.axis("tight")
            plt.xlabel('GPA')
            plt.ylabel('Salary')
            plt.legend()
            plt.title(f"KNeighborsRegressor: GPA (k = {neighbor}, weights = {weights})")

            plt.subplot(2, 2, 3)
            plt.scatter(X_train.relatedness_to_career_goals, y_train, color="darkorange", label="data")
            plt.plot(X_test.relatedness_to_career_goals, y_pred, color="navy", label="prediction")
            plt.axis("tight")
            plt.xlabel('Relatedness to career')
            plt.ylabel('Salary')
            plt.legend()
            plt.title(f"KNeighborsRegressor: Relatedness to career (k = {neighbor}, weights = {weights})")

            plt.subplot(2, 2, 4)
            plt.scatter(X_train.number_of_internships, y_train, color="darkorange", label="data")
            plt.plot(X_test.number_of_internships, y_pred, color="navy", label="prediction")
            plt.axis("tight")
            plt.xlabel('Number of Internships')
            plt.ylabel('Salary')
            plt.legend()
            plt.title(f"KNeighborsRegressor: Number of Internships (k = {neighbor}, weights = {weights})")

            plt.tight_layout()
            plt.show()



def get_max_r2(major, X_train, X_test, y_train, y_test):
    #max_acc: dict[Any, float] = {}
    max_acc = []

    plt.figure(figsize=(14, 6))
    plt.suptitle(major)
    for i, weights in enumerate(["uniform", "distance"]):
        train_score: dict[Any, float] = {}
        test_score: dict[Any, float] = {}
        n_neighbors = np.arange(2, 30, 1)
        for neighbor in n_neighbors:
            knn = KNeighborsRegressor(n_neighbors=neighbor, weights=weights)
            knn.fit(X_train, y_train)
            train_score[neighbor] = knn.score(X_train, y_train)
            test_score[neighbor] = knn.score(X_test, y_test)

        plt.subplot(1, 2, i + 1)
        plt.plot(n_neighbors, train_score.values(), label="Train Accuracy")
        plt.plot(n_neighbors, test_score.values(), label="Test Accuracy")
        plt.xlabel("Number Of Neighbors")
        plt.ylabel("Accuracy")
        plt.title("KNN: Varying number of Neighbors: weights = '%s')" % (weights))
        plt.legend()
        #plt.xlim(0, 33)
        #plt.ylim(0.60, 0.90)
        plt.grid()


        for key, value in test_score.items():
            if value == max(test_score.values()):
                print(f"Best Accuracy Score is {round(value, 2)} with {key} neighbors. With weights as {weights}")
                #max_acc[key] = value
                max_acc.append(key)

    plt.show()
    return max_acc

def get_para_info(X, y):
    print("No Exclusions")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=40)
    knn_regressor = KNeighborsRegressor(n_neighbors=4)

    knn_regressor.fit(X_train, y_train)

    y_pred = knn_regressor.predict(X_test)

    r2 = round(r2_score(y_test, y_pred), 2)
    print(f"R-squared: {r2}")

    """accuracy = round(knn_regressor.score(X_test, y_test), 2)
    print(f"Accuracy: {accuracy}, for weights = {weights}")"""

    mse = round(mean_squared_error(y_test, y_pred), 2)
    print(f"Mean Squared Error: {mse}")
    print("------------------------")
    for i in X.columns.values:
        print(f"Exclude {i}")
        tempX = X.drop(i, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(tempX, y, test_size=0.22, random_state=40)
        knn_regressor = KNeighborsRegressor(n_neighbors=4)

        knn_regressor.fit(X_train, y_train)

        y_pred = knn_regressor.predict(X_test)

        r2 = round(r2_score(y_test, y_pred), 2)
        print(f"R-squared: {r2}")

        """accuracy = round(knn_regressor.score(X_test, y_test), 2)
        print(f"Accuracy: {accuracy}, for weights = {weights}")"""

        mse = round(mean_squared_error(y_test, y_pred), 2)
        print(f"Mean Squared Error: {mse}")
        print("------------------------")

def k_nearest_neighbors():
    data = pd.read_csv('csi_alum_data.csv')
    print("Resluts for CSI Alum")
    X = data.drop('salary', axis=1)  # Drop the 'salary' column to get the feature set
    y = data['salary']  # Target variable
    #get_para_info(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=40)
    max_acc = get_max_r2("CSI", X_train, X_test, y_train, y_test)
    get_info("CSI", max_acc, X_train, X_test, y_train, y_test)
    print("------------------------------")
    data = pd.read_csv('cncs_alum_data.csv')
    print("Resluts for CNCS Alum")
    X = data.drop('salary', axis=1)  # Drop the 'salary' column to get the feature set
    y = data['salary']  # Target variable
    #get_para_info(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=40)
    max_acc = get_max_r2("CNCS",X_train, X_test, y_train, y_test)
    get_info("CNCS", max_acc, X_train, X_test, y_train, y_test)
    print("------------------------------")
    data = pd.read_csv('cdf_alum_data.csv')
    print("Resluts for CDF Alum")
    X = data.drop('salary', axis=1)  # Drop the 'salary' column to get the feature set
    y = data['salary']  # Target variable
    #get_para_info(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=40)
    max_acc = get_max_r2("CDF", X_train, X_test, y_train, y_test)
    get_info("CDF", max_acc, X_train, X_test, y_train, y_test)
    print("------------------------------")


if __name__ == '__main__':
    k_nearest_neighbors()
