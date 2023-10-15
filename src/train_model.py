import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
import pickle

# Download S&P500 data from Yahoo Finance
data = yf.download('^GSPC', start='2000-01-01', end='2023-01-01')
prices = data['Close'].values.reshape(-1, 1)

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)

# Creating a dataset with 60 timesteps and 1 output
X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)

# Reshape the data to fit the RNN input shape
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define the RNN architecture
model = Sequential()
model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(SimpleRNN(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model. With more than 30 epochs the loss does not improve much
model.fit(X_train, y_train, epochs=30, batch_size=32)

# Save the model and scaler for future use
model.save('./assets/models/sp500_rnn_model.h5')
with open('./assets/models/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
