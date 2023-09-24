import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

def train_and_evaluate_gb():
    # Load processed data
    data = pd.read_csv('datasets/processed_data.csv')

    # Separate features and target variable
    X = data.drop('CallLength', axis=1)
    y = data['CallLength']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Gradient Boosting Model
    gb = GradientBoostingRegressor(random_state=42)

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

    # Fit model
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f'Best Parameters: {best_params}')

    # Evaluate the model with the best parameters on the test set
    best_gb = grid_search.best_estimator_
    y_pred = best_gb.predict(X_test)
    print(f'Mean Squared Error on test set: {mean_squared_error(y_test, y_pred)}')
    print(f'R^2 Score on test set: {r2_score(y_test, y_pred)}')

if __name__ == "__main__":
    train_and_evaluate_gb()

