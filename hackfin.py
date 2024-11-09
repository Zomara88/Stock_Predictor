import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Fetch data
data = yf.Ticker("TSLA")
hist = data.history(period="max")
actions = data.actions  # Includes dividends and splits

# Feature Engineering
hist['Dividends'] = hist['Dividends'] > 0
hist['Stock Splits'] = hist['Stock Splits'] > 0
hist['5-Day MA'] = hist['Close'].rolling(window=5).mean()
hist['10-Day MA'] = hist['Close'].rolling(window=10).mean()
hist['Lagged Close'] = hist['Close'].shift(1)
hist.dropna(inplace=True)  # Remove NaNs

# Prepare data
features = ['Lagged Close', '5-Day MA', '10-Day MA', 'Dividends', 'Stock Splits', 'Volume']
X = hist[features]
y = hist['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Fit models and make predictions
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    results[name] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
        'MAE': mean_absolute_error(y_test, predictions),
        'MAPE': np.mean(np.abs((y_test - predictions) / y_test)) * 100,
        'R2 Score': r2_score(y_test, predictions)
    }

# Print model performance metrics
for name, metrics in results.items():
    print(f"{name} Model Performance:")
    print(f"RMSE: {metrics['RMSE']}")
    print(f"MAE: {metrics['MAE']}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"R2 Score: {metrics['R2 Score']:.4f}\n")
