import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load processed data
data = pd.read_csv('datasets/processed_data.csv')

# Separate features and target variable
X = data.drop(['CallLength'], axis=1)
y = data['CallLength']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Model with the best parameters obtained from RandomizedSearchCV
best_rf = RandomForestRegressor(n_estimators=100, min_samples_split=2, min_samples_leaf=4, max_depth=10,
                                random_state=42)

# Fit model
best_rf.fit(X_train, y_train)

# Make predictions and evaluate on the test set
y_pred = best_rf.predict(X_test)
print(f'Mean Squared Error on test set: {mean_squared_error(y_test, y_pred)}')
print(f'R^2 Score on test set: {r2_score(y_test, y_pred)}')

# Extract feature importances and plot
feature_importances = best_rf.feature_importances_
features_list = list(X.columns)

# Create DataFrame to hold features and their importances
feature_importance_df = pd.DataFrame({'Feature': features_list, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

