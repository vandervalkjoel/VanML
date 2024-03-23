import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_and_evaluate():
    # Load processed data
    data = pd.read_csv('datasets/processed_data.csv')

    # Separate features and target variable, and drop 'Number_of_Referrals' from features
    X = data.drop(['CallLength'], axis=1)
    y = data['CallLength']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Add a constant to the features (required for statsmodels GLM)
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    # Initialize GLM Model (using Gaussian family as an example, you might want to choose a different family
    # depending on your response variable distribution and link function)
    glm = sm.GLM(y_train, X_train_const, family=sm.families.Gaussian())

    # Fit model
    glm_results = glm.fit()

    # Display feature importances as percentages
    coefficients = glm_results.params.drop('const')
    sum_of_coefficients = np.sum(coefficients)
    relative_importance = (coefficients / sum_of_coefficients) * 100  # converting to percentage

    feature_importance = pd.DataFrame({'Feature': coefficients.index,
                                       'Relative Importance (%)': relative_importance})
    feature_importance = feature_importance.sort_values(by='Relative Importance (%)', ascending=False)
    print("Feature Importance:")
    print(feature_importance)

    # Make predictions and evaluate on the test set
    y_pred = glm_results.predict(X_test_const)
    print(f'Mean Squared Error on test set: {mean_squared_error(y_test, y_pred)}')
    print(f'R^2 Score on test set: {r2_score(y_test, y_pred)}')

if __name__ == "__main__":
    train_and_evaluate()