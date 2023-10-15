import numpy as np
import matplotlib.pyplot as plt
import pickle
import yfinance as yf
from tensorflow.keras.models import load_model

# Load the saved model and scaler
model = load_model('./assets/models/sp500_rnn_model.h5')
with open('./assets/models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Download S&P500 data from Yahoo Finance
data = yf.download('^GSPC', start='2000-01-01', end='2023-01-01')
prices = data['Close'].values.reshape(-1, 1)

# Preprocess the data
scaled_data = scaler.transform(prices)

# Creating a dataset with 60 timesteps and 1 output
X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Predict using the model
predicted_prices = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))

# Plotting actual vs predicted prices
plt.figure(figsize=(14,5))
plt.plot(data['Close'].index[60:], prices[60:], color='red', label='Real S&P500 Price')
plt.plot(data['Close'].index[60:], predicted_prices, color='blue', label='Predicted S&P500 Price')
plt.title('S&P500 Price Prediction')
plt.xlabel('Time')
plt.ylabel('S&P500 Price')
plt.legend()
plt.show()
