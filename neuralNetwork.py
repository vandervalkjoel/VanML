import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load processed data
data = pd.read_csv('datasets/processed_data.csv')

# Separate features and target variable
X = data.drop('CallLength', axis=1)
y = data['CallLength']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Sequential model
model = Sequential()

# Add input layer with input_dim equal to the number of features
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))

# Add hidden layers
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

# Add output layer
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Make predictions and evaluate on the test set
y_pred = model.predict(X_test_scaled).flatten()
print(f'Mean Squared Error on test set: {mean_squared_error(y_test, y_pred)}')
print(f'R^2 Score on test set: {r2_score(y_test, y_pred)}')