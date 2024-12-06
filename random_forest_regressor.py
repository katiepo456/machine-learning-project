from sklearn.metrics import mean_squared_error
import sklearn.tree
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.ensemble
from sklearn.model_selection import train_test_split


def plot_model(model, df):
    # Try to plot, if we cant, display data
    try:
        tree_to_plot = model.estimators_[0]

        print("...Printing graph...")

        plt.figure(figsize=(10,20))
        sklearn.tree.plot_tree(tree_to_plot, feature_names=df.drop(['salary'], axis='columns').columns.tolist(), filled=True, rounded=True, fontsize=5)
        plt.title(f"Decision Tree ({model.__class__.__name__})")
        plt.show()

    except:
        print("!" * 50)
        print("Model does not have a tree to plot")
        print("!" * 50)
        print(f"\nModel: {model.__class__.__name__}")
        print(f"Optimal Params: {model.best_params_}")
        print(f"Best Score: {model.best_score_}")
        print(f"Features: {model.best_estimator_.feature_names_in_}")
        print(f"Feature Importances: {model.best_estimator_.feature_importances_}")

def tune_parameters(X_train, y_train):
    param_grid = {
        'n_estimators': [200],
        'max_depth': [5, 6],
        'max_features': [2, 3, 4],
        'max_leaf_nodes': [5, 10, 15],
        'min_samples_leaf': [1, 2],
    }

    rfr_cv = sklearn.model_selection.GridSearchCV(sklearn.ensemble.RandomForestRegressor(random_state=0), param_grid, cv=3, n_jobs=-1)

    return rfr_cv.fit(X_train, y_train)

def test_and_graph(models, X_test, y_test, df):
    for model in models:
        print("=" * 50)
        print(f"Model: {model.__class__.__name__}")
        print(f"R Squared Score: {model.score(X_test, y_test)}")
        print(f"MSE: {mean_squared_error(y_test, model.predict(X_test))}")
        plot_model(model, df)

def random_forest_model():
    csi_df = pd.read_csv('csi_alum_data.csv')
    cncs_df = pd.read_csv('cncs_alum_data.csv')
    cdf_df = pd.read_csv('cdf_alum_data.csv')

    df = pd.concat([csi_df, cncs_df, cdf_df])

    #['salary', 'time_frame_for_securing_position', 'relatedness_to_career_goals', 'study_abroad?', 'student_leadership_position?', 'center_of_experience?', 'number_of_internships', 'GPA'

    drops = ['time_frame_for_securing_position', 'student_leadership_position?', 'center_of_experience?']

    df = df.drop(drops, axis='columns')

    model = sklearn.ensemble.RandomForestRegressor(n_estimators=250, max_depth=5, max_features=2, max_leaf_nodes=5, min_samples_leaf=1, random_state=0)
    
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['salary'], axis='columns'), df.salary, test_size=0.2)

    model = model.fit(X_train, y_train)

    rfr_tuned = tune_parameters(X_train, y_train)

    test_and_graph([model, rfr_tuned], X_test, y_test, df)

if __name__ == "__main__":
    random_forest_model()