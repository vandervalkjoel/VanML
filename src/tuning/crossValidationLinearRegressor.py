import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_and_evaluate():
    # Load processed data
    data = pd.read_csv('datasets/processed_data.csv')

    # Separate features and target variable, and drop 'Number_of_Referrals' from features
    X = data.drop(['CallLength', 'Number_of_Referrals'], axis=1)
    y = data['CallLength']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Linear Regression Model
    lr = LinearRegression()

    # Perform Cross-validation
    scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

    # Negative MSE scores
    print(f'Negative Mean Squared Error scores: {scores}')

    # Mean and standard deviation of the scores
    print(f'Mean: {np.mean(scores)}, Standard Deviation: {np.std(scores)}')

    # Train Linear Regression Model on the whole training set
    lr.fit(X_train, y_train)

    # Display feature importances as percentages
    coefficients = lr.coef_
    absolute_coefficients = np.abs(coefficients)
    sum_of_absolute_coefficients = np.sum(absolute_coefficients)
    relative_importance = (absolute_coefficients / sum_of_absolute_coefficients) * 100  # converting to percentage

    feature_importance = pd.DataFrame({'Feature': X_train.columns,
                                       'Relative Importance (%)': relative_importance})
    feature_importance = feature_importance.sort_values(by='Relative Importance (%)', ascending=False)
    print("Feature Importance:")
    print(feature_importance)

    # Make predictions and evaluate on the test set
    y_pred = lr.predict(X_test)
    print(f'Mean Squared Error on test set: {mean_squared_error(y_test, y_pred)}')
    print(f'R^2 Score on test set: {r2_score(y_test, y_pred)}')

if __name__ == "__main__":
    train_and_evaluate()

# Negative Mean Squared Error scores: [-1.36085448e+15 -3.59878980e+01 -3.50855177e+01 -4.86913560e+01
#  -5.04631682e+01]
# Mean: -272170896467227.4, Standard Deviation: 544341792934369.7
# Mean Squared Error on test set: 33.90174981386552
# R^2 Score on test set: 0.27913459101953975