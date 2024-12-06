import sklearn.tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.ensemble


def random_forest():
    df = pd.read_csv('csi_alum_data.csv')

    x = df.iloc[:, 1:8].values  # Features
    y = df.iloc[:, 0].values  # Target Variable

    regressor = sklearn.ensemble.RandomForestRegressor(max_leaf_nodes=20, random_state=0, oob_score=True)

    regressor.fit(x, y)

    print(f"Out-of-bag score: {regressor.oob_score_}")

    predictions = regressor.predict(x)

    mse = sklearn.metrics.mean_squared_error(y, predictions)
    print(f"Mean squared error: {mse}")

    r2 = sklearn.metrics.r2_score(y, predictions)
    print(f"R2 score: {r2}")

    tree_to_plot = regressor.estimators_[0]

    plt.figure(figsize=(10,8))
    sklearn.tree.plot_tree(tree_to_plot, feature_names=df.iloc[:,1:8].columns.tolist(), filled=True, rounded=True, fontsize=5)
    plt.title("Decision Tree")
    plt.show()


if __name__ == '__main__':
    random_forest()
