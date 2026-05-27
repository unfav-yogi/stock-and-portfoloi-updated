import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

stocks = ['AAPL', 'TSLA', 'GOOG', 'MSFT']

data = yf.download(
    stocks,
    start='2020-01-01',
    end='2024-01-01'
)

close_prices = data['Close']

print("\nCLOSING PRICES DATA\n")
print(close_prices.head())

daily_returns = close_prices.pct_change()

daily_returns = daily_returns.dropna()

print("\nDAILY RETURNS\n")
print(daily_returns.head())

average_returns = daily_returns.mean()

print("\nAVERAGE RETURNS OF EACH STOCK\n")
print(average_returns)

risk = daily_returns.std()

print("\nRISK OF EACH STOCK\n")
print(risk)

portfolio_weights = np.array([0.25, 0.25, 0.25, 0.25])

portfolio_return = np.sum(
    average_returns * portfolio_weights
)

portfolio_risk = np.sqrt(
    np.dot(
        portfolio_weights.T,
        np.dot(
            daily_returns.cov(),
            portfolio_weights
        )
    )
)

print("\nPORTFOLIO RETURN\n")
print(portfolio_return)

print("\nPORTFOLIO RISK\n")
print(portfolio_risk)

plt.figure(figsize=(14, 7))

for stock in close_prices.columns:
    plt.plot(
        close_prices[stock],
        label=stock
    )

plt.title("Stock Price Comparison")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.legend()

plt.show()

plt.figure(figsize=(10, 6))

plt.bar(
    average_returns.index,
    average_returns.values
)

plt.title("Average Returns of Stocks")
plt.xlabel("Stocks")
plt.ylabel("Average Return")

plt.show()

plt.figure(figsize=(10, 6))

plt.bar(
    risk.index,
    risk.values
)

plt.title("Risk Analysis of Stocks")
plt.xlabel("Stocks")
plt.ylabel("Risk")

plt.show()

X = np.arange(len(close_prices)).reshape(-1, 1)

y = close_prices['AAPL'].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print("\nLINEAR REGRESSION RESULTS\n")

print("Mean Squared Error :", mse)

print("R2 Score :", r2)

comparison = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': predictions
})

print("\nACTUAL VS PREDICTED\n")
print(comparison.head(20))

plt.figure(figsize=(14, 7))

plt.plot(
    y_test,
    label='Actual Prices'
)

plt.plot(
    predictions,
    label='Predicted Prices'
)

plt.title("Apple Stock Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()

plt.show()

allocation = pd.DataFrame({
    'Stock': stocks,
    'Weight': portfolio_weights
})

print("\nPORTFOLIO ALLOCATION\n")
print(allocation)

plt.figure(figsize=(8, 8))

plt.pie(
    allocation['Weight'],
    labels=allocation['Stock'],
    autopct='%1.1f%%'
)

plt.title("Portfolio Distribution")

plt.show()

print("\nPROJECT ANALYSIS COMPLETED SUCCESSFULLY\n")