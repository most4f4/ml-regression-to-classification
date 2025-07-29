import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


dataset = pd.read_csv('house_price.csv')
print("Dataset loaded successfully.")

# Check for non-numeric values or missing values
print(dataset.dtypes)
print(dataset.isnull().sum())

# Assuming 'price' is the target variable and the rest are features
X = dataset.drop("price", axis=1)
y = dataset["price"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("="*60)
print("LINEAR REGRESSION")
print("="*60)

# Train the Linear Regression model
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

# Get the slope and intercept of the regression lin
slope = regression_model.coef_
intercept = regression_model.intercept_

print(f"Slope: {slope}")
print(f"Intercept: {intercept}")

# Make predictions on the test set
y_pred = regression_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {np.sqrt(mse)}")
print(f"Mean Absolute Percentage Error: {mape}")

print("="*60)
print("STOCHASTIC GRADIENT DESCENT REGRESSION")
print("="*60)

# Train the SGD Regressor model
sgd_model = SGDRegressor()
sgd_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_sgd = sgd_model.predict(X_test)

# Evaluate the SGD model
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
mae_sgd = mean_absolute_error(y_test, y_pred_sgd)
mape_sgd = mean_absolute_percentage_error(y_test, y_pred_sgd)

print(f"Mean Squared Error: {mse_sgd}")
print(f"Mean Absolute Error: {mae_sgd}")
print(f"Root Mean Squared Error: {np.sqrt(mse_sgd)}")
print(f"Mean Absolute Percentage Error: {mape_sgd}")