import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import yfinance as yf

stock_data = yf.download(
    "AAPL",
    start="2020-01-01",
    end="2024-01-01"
)

stock_data.columns = stock_data.columns.get_level_values(0)

stock_data.reset_index(inplace=True)

print("\nFIRST 10 ROWS OF DATA\n")
print(stock_data.head(10))

print("\nDATA INFORMATION\n")
print(stock_data.info())

print("\nSTATISTICAL SUMMARY\n")
print(stock_data.describe())

print("\nNULL VALUES\n")
print(stock_data.isnull().sum())

stock_data['Moving_Average_10'] = (
    stock_data['Close'].rolling(window=10).mean()
)

stock_data['Moving_Average_50'] = (
    stock_data['Close'].rolling(window=50).mean()
)

stock_data['Daily_Return'] = (
    stock_data['Close'].pct_change()
)

stock_data.dropna(inplace=True)

print("\nUPDATED DATA\n")
print(stock_data.head())

X = stock_data[[
    'Open',
    'High',
    'Low',
    'Volume',
    'Moving_Average_10',
    'Moving_Average_50'
]]

y = stock_data['Close']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(X_train, y_train)

rf_predictions = random_forest_model.predict(X_test)

linear_mse = mean_squared_error(
    y_test,
    linear_predictions
)

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

rf_mse = mean_squared_error(
    y_test,
    rf_predictions
)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

print("\nLINEAR REGRESSION RESULTS\n")
print("Mean Squared Error :", linear_mse)
print("Mean Absolute Error :", linear_mae)
print("R2 Score :", linear_r2)

print("\nRANDOM FOREST RESULTS\n")
print("Mean Squared Error :", rf_mse)
print("Mean Absolute Error :", rf_mae)
print("R2 Score :", rf_r2)

comparison = pd.DataFrame({
    'Actual Price': y_test.values,
    'Linear Prediction': linear_predictions,
    'Random Forest Prediction': rf_predictions
})

print("\nCOMPARISON TABLE\n")
print(comparison.head(20))

plt.figure(figsize=(14, 7))

plt.plot(
    y_test.values,
    label='Actual Price'
)

plt.plot(
    linear_predictions,
    label='Linear Regression Prediction'
)

plt.plot(
    rf_predictions,
    label='Random Forest Prediction'
)

plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.legend()

plt.show()

plt.figure(figsize=(14, 7))

plt.plot(
    stock_data.index,
    stock_data['Close'],
    label='Closing Price'
)

plt.plot(
    stock_data.index,
    stock_data['Moving_Average_10'],
    label='10 Day Moving Average'
)

plt.plot(
    stock_data.index,
    stock_data['Moving_Average_50'],
    label='50 Day Moving Average'
)

plt.title("Stock Closing Price and Moving Averages")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()

plt.show()

plt.figure(figsize=(12, 6))

plt.hist(
    stock_data['Daily_Return'],
    bins=50
)

plt.title("Distribution of Daily Returns")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")

plt.show()

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': random_forest_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFEATURE IMPORTANCE\n")
print(feature_importance)

plt.figure(figsize=(10, 5))

plt.bar(
    feature_importance['Feature'],
    feature_importance['Importance']
)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.show()

new_data = pd.DataFrame({
    'Open': [180],
    'High': [185],
    'Low': [178],
    'Volume': [50000000],
    'Moving_Average_10': [182],
    'Moving_Average_50': [175]
})

future_prediction_linear = linear_model.predict(new_data)

future_prediction_rf = random_forest_model.predict(new_data)

print("\nFUTURE STOCK PRICE PREDICTION\n")

print(
    "Linear Regression Predicted Price :",
    future_prediction_linear[0]
)

print(
    "Random Forest Predicted Price :",
    future_prediction_rf[0]
)

plt.figure(figsize=(8, 5))

models = [
    'Linear Regression',
    'Random Forest'
]

scores = [
    linear_r2,
    rf_r2
]

plt.bar(models, scores)

plt.title("Model Accuracy Comparison")
plt.ylabel("R2 Score")

plt.show()

plt.figure(figsize=(14, 7))

plt.scatter(
    y_test,
    linear_predictions
)

plt.title("Actual vs Predicted Scatter Plot")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.show()

print("\nPROJECT COMPLETED SUCCESSFULLY\n")
