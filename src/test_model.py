import pickle

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

from common import (MODEL_PATH, PLOT_PATH, SCALER_PATH, download_prices,
                    make_windows, split_index)

# Load the saved model and scaler
if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    raise SystemExit(f'Model not found in {MODEL_PATH.parent}. Run train_model.py first.')
model = load_model(str(MODEL_PATH))
with open(SCALER_PATH, 'rb') as file:
    scaler = pickle.load(file)

# Download S&P500 data from Yahoo Finance
close = download_prices()
prices = close.values.reshape(-1, 1)

# Use the same split as train_model.py: only the test period is evaluated,
# since the model has already seen the training period
split = split_index(len(prices))
scaled_data = scaler.transform(prices)
X_test, _ = make_windows(scaled_data, split, len(scaled_data))

# Predict using the model
predicted_prices = scaler.inverse_transform(model.predict(X_test).reshape(-1, 1))[:, 0]
real_prices = prices[split:, 0]
dates = close.index[split:]

# Baseline: predict that each day closes at the previous day's price. Each
# prediction uses the real prices of the previous days, so a useful model
# must beat this baseline, not just follow the real curve.
baseline_prices = prices[split - 1:-1, 0]


def print_metrics(name, predicted):
    errors = predicted - real_prices
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors / real_prices)) * 100
    print(f'{name:<28} RMSE: {rmse:9.2f}   MAE: {mae:9.2f}   MAPE: {mape:6.2f}%')


print(f'Test period: {dates[0].date()} to {dates[-1].date()} ({len(dates)} days)')
print_metrics('RNN model', predicted_prices)
print_metrics('Baseline (previous close)', baseline_prices)

# Plotting actual vs predicted prices
plt.figure(figsize=(14, 5))
plt.plot(dates, real_prices, color='red', label='Real S&P500 Price')
plt.plot(dates, predicted_prices, color='blue', label='Predicted S&P500 Price')
plt.title('S&P500 Price Prediction (test period)')
plt.xlabel('Time')
plt.ylabel('S&P500 Price')
plt.legend()
plt.savefig(PLOT_PATH)
print(f'Plot saved to {PLOT_PATH}')
plt.show()
