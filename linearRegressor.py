import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_and_evaluate():
    # Load processed data
    data = pd.read_csv('datasets/processed_data.csv')

    # Separate features and target variable
    X = data.drop('CallLength', axis=1)
    y = data['CallLength']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression Model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = lr.predict(X_test)
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
    print(f'R^2 Score: {r2_score(y_test, y_pred)}')

if __name__ == "__main__":
    train_and_evaluate()