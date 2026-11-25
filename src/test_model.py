import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

from common import (MODEL_PATH, NORMALIZATION_PATH, PLOT_PATH, Normalizer,
                    load_dataset, make_windows, predict_returns, split_index)
from evaluation import evaluate, print_report

# Load the saved model and normalization
if not MODEL_PATH.exists() or not NORMALIZATION_PATH.exists():
    raise SystemExit(f'Model not found in {MODEL_PATH.parent}. Run train_model.py first.')
model = load_model(str(MODEL_PATH))
normalizer = Normalizer.load(NORMALIZATION_PATH)

# Download the S&P 500 and the other variables from Yahoo Finance
features, close, previous_close = load_dataset()

# Use the same split as train_model.py: only the test period is evaluated,
# since the model has already seen the training period
split = split_index(len(features))
z = normalizer.transform(features)
X_test, _ = make_windows(z, normalizer.target_index, split, len(z))

# The model predicts the return of each day; the price is the previous close
# moved by that return
previous = previous_close.values[split:]
predicted_prices = previous * np.exp(predict_returns(model, normalizer, X_test))
real_prices = close.values[split:]
dates = close.index[split:]

print(f'Test period: {dates[0].date()} to {dates[-1].date()} ({len(dates)} days)')
print_report(evaluate(real_prices, previous, predicted_prices))

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
