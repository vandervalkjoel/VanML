from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

def linear_model():
    # Load processed data
    data = pd.read_csv('datasets/processed_data.csv')

    # Separate features and target variable
    X = data.drop(['CallLength'], axis=1)
    y = data['CallLength']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Linear Regression Model
    lr = LinearRegression()

    # Train Linear Regression Model on the training set
    lr.fit(X_train, y_train)

    # Display coefficients for each feature
    coef_dict = dict(zip(X_train.columns, lr.coef_))
    print("Coefficients for each feature:")
    for feature, coef in coef_dict.items():
        print(f"{feature}: {coef}")

    # Make predictions and evaluate on the test set
    y_pred = lr.predict(X_test)
    print(f'\nMean Squared Error on test set: {mean_squared_error(y_test, y_pred)}')
    print(f'R^2 Score on test set: {r2_score(y_test, y_pred)}')

if __name__ == "__main__":
    linear_model()