import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sentiment import get_sentiment_data  # Ensure you have a correct path for importing
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Constants
STOCK_TICKER = 'NVDA'
NEWS_API_KEY = 'fe2cb004c56947e2b0a25259cef6ac44'

# Fetch data
data = yf.Ticker(STOCK_TICKER)
hist = data.history(period="max")

# Prepare historical data by making sure the index is a column and dates are timezone-naive
hist.reset_index(inplace=True)
hist['Date'] = hist['Date'].dt.tz_localize(None)

# Incorporate Sentiment Data
sentiment_data = get_sentiment_data(STOCK_TICKER, NEWS_API_KEY)
sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date']).dt.tz_localize(None)  # Ensure tz-naive

# Now merge with financial data
combined_data = pd.merge(hist, sentiment_data, on='Date', how='left')
combined_data.ffill(inplace=True)  # Use forward fill to handle non-trading days

# Create lagged feature for 'Close' price
combined_data['Previous_Close'] = combined_data['Close'].shift(1)

# Remove the first row which will have NaN values due to shifting
combined_data = combined_data.dropna(subset=['Previous_Close', 'Sentiment'])

# Define features and target
features = ['Previous_Close', 'Sentiment']
y = combined_data['Close']
X = combined_data[features]

# Split the data into train and test sets sequentially
n = len(combined_data)
train_pct = 0.8
split_idx = int(n * train_pct)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'Linear Regression': LinearRegression(),
    'SVR': SVR(C=1000, gamma='scale')
}

# Initialize Neural Network
nn_model = Sequential([
    Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Fit models and make predictions
results = {}
predictions_dict = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    predictions_dict[name] = predictions
    results[name] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
        'MAE': mean_absolute_error(y_test, predictions),
        'MAPE': np.mean(np.abs((y_test - predictions) / y_test)) * 100,
        'R2 Score': r2_score(y_test, predictions)
    }

# Fit and predict with neural network
nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
nn_predictions = nn_model.predict(X_test_scaled).flatten()
predictions_dict['Neural Network'] = nn_predictions
results['Neural Network'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, nn_predictions)),
    'MAE': mean_absolute_error(y_test, nn_predictions),
    'MAPE': np.mean(np.abs((y_test - nn_predictions) / y_test)) * 100,
    'R2 Score': r2_score(y_test, nn_predictions)
}

# Print model performance metrics and plot results
for name, metrics in results.items():
    print(f"{name} Model Performance:")
    print(f"RMSE: {metrics['RMSE']}")
    print(f"MAE: {metrics['MAE']}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"R2 Score: {metrics['R2 Score']:.4f}\n")
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values[:50], label='Actual Prices', color='blue', marker='o')
    plt.plot(predictions_dict[name][:50], label='Predicted Prices', color='red', linestyle='--', marker='x')
    plt.title(f'Comparison of Actual and Predicted Prices: {name}')
    plt.xlabel('Samples')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
