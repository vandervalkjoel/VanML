import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_and_evaluate():
    # Load processed data
    data = pd.read_csv('datasets/processed_data.csv')

    # Drop 'Number_of_Referrals' feature
    X = data.drop(['CallLength', 'Number_of_Referrals'], axis=1)
    y = data['CallLength']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define parameter grid for Random Forest
    rf_param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform Randomized Search for Random Forest
    rf_random_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_distributions=rf_param_grid,
                                          n_iter=10, cv=3, scoring='r2', n_jobs=-1, random_state=42)
    rf_random_search.fit(X_train, y_train)
    print(f'Best Random Forest Parameters: {rf_random_search.best_params_}')
    print(f'Best Random Forest R^2 Score: {rf_random_search.best_score_}')

    # Define parameter grid for Gradient Boosting
    gb_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Perform Randomized Search for Gradient Boosting
    gb_random_search = RandomizedSearchCV(GradientBoostingRegressor(random_state=42), param_distributions=gb_param_grid,
                                          n_iter=10, cv=3, scoring='r2', n_jobs=-1, random_state=42)
    gb_random_search.fit(X_train, y_train)
    print(f'Best Gradient Boosting Parameters: {gb_random_search.best_params_}')
    print(f'Best Gradient Boosting R^2 Score: {gb_random_search.best_score_}')

    # Evaluate the best Random Forest model
    rf_best_model = rf_random_search.best_estimator_
    y_pred_rf = rf_best_model.predict(X_test)
    print(f'Random Forest Mean Squared Error: {mean_squared_error(y_test, y_pred_rf)}')
    print(f'Random Forest R^2 Score: {r2_score(y_test, y_pred_rf)}')

    # Evaluate the best Gradient Boosting model
    gb_best_model = gb_random_search.best_estimator_
    y_pred_gb = gb_best_model.predict(X_test)
    print(f'Gradient Boosting Mean Squared Error: {mean_squared_error(y_test, y_pred_gb)}')
    print(f'Gradient Boosting R^2 Score: {r2_score(y_test, y_pred_gb)}')


if __name__ == "__main__":
    train_and_evaluate()