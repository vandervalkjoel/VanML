import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
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

    # Initialize and fit Decision Tree Model with the best parameters
    dt = DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, min_samples_split=2, random_state=42)
    dt.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = dt.predict(X_test)
    print(f'Mean Squared Error on test set: {mean_squared_error(y_test, y_pred)}')
    print(f'R^2 Score on test set: {r2_score(y_test, y_pred)}')

    # Extract and display feature importances in descending order
    feature_importances = dt.feature_importances_
    features_list = list(X.columns)
    feature_importance_df = pd.DataFrame({'Feature': features_list, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)  # Top 10 features

    # Plotting the Feature Importances with gradient color
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Oranges(np.linspace(1.0, 0.4, len(feature_importance_df['Feature'])))  # Gradient color
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=colors)
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importances')
    plt.gca().invert_yaxis()  # Display the feature with the highest importance at the top
    plt.tight_layout()  # Ensure everything fits without overlapping
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()