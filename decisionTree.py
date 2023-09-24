import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_and_evaluate():
    # Load processed data
    data = pd.read_csv('datasets/processed_data.csv')

    # Separate features and target variable
    X = data.drop(['CallLength'], axis=1)
    y = data['CallLength']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Decision Tree Model
    dt = DecisionTreeRegressor(random_state=42)

    # Define hyperparameter grid
    param_grid = {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

    # Fit model
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f'Best Parameters: {best_params}')

    # Evaluate the model with the best parameters on the test set
    best_dt = grid_search.best_estimator_
    y_pred = best_dt.predict(X_test)
    print(f'Mean Squared Error on test set: {mean_squared_error(y_test, y_pred)}')
    print(f'R^2 Score on test set: {r2_score(y_test, y_pred)}')

    # Extract and display feature importances in descending order
    feature_importances = best_dt.feature_importances_
    features_list = list(X.columns)
    feature_importance_df = pd.DataFrame({'Feature': features_list, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)  # Top 10 features

    # Plotting the Feature Importances
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'],
             color=np.linspace(0.2, 1.0, len(feature_importance_df['Feature'])), cmap='Blues_r')  # Gradient color
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importances')
    plt.gca().invert_yaxis()  # Display the feature with the highest importance at the top
    plt.tight_layout()  # Ensure everything fits without overlapping
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()


# Output: Fitting 5 folds for each of 96 candidates, totalling 480 fits
# Best Parameters: {'max_depth': 10, 'min_samples_leaf': 10, 'min_samples_split': 2}
# Mean Squared Error on test set: 31.860127368219878
# R^2 Score on test set: 0.3225463620149739