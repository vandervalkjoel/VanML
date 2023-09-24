import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load processed data
data = pd.read_csv('datasets/processed_data.csv')

# Separate features and target variable
X = data.drop('CallLength', axis=1)
y = data['CallLength']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit a RandomForestRegressor to extract feature importances
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Extract feature importances and select the top 10 features
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
top_features = feature_importances.sort_values(by='Importance', ascending=False).head(10)['Feature'].tolist()


# Select only the top 10 features for X_train and X_test
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

# Define hyperparameter grid
param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(rf, param_distributions=param_distributions, n_iter=10, cv=5,
                                   scoring='neg_mean_squared_error', n_jobs=-1, verbose=1, random_state=42)

# Fit model
random_search.fit(X_train_selected, y_train)

# Get the best parameters and the best score
best_params = random_search.best_params_
best_score = random_search.best_score_
print(f'Best Parameters: {best_params}')
print(f'Best Cross-Validated Score on Training Data: {best_score}')

# Evaluate the model with the best parameters on the test set
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test_selected)
print(f'Mean Squared Error on test set: {mean_squared_error(y_test, y_pred)}')
print(f'R^2 Score on test set: {r2_score(y_test, y_pred)}')
