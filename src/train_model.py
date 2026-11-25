import pickle

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout

from common import (MODEL_PATH, MODELS_DIR, SCALER_PATH, SEED, WINDOW,
                    download_prices, make_windows, split_index)

# Make the results reproducible
tf.keras.utils.set_random_seed(SEED)

# Create the output folder before training, so saving cannot fail at the end
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Download S&P500 data from Yahoo Finance
prices = download_prices().values.reshape(-1, 1)

# Chronological split: the most recent days are kept for testing
split = split_index(len(prices))

# Fit the scaler on the training period only, so no information from the
# test period leaks into the model
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(prices[:split])
scaled_data = scaler.transform(prices)

# Each sample uses the previous 60 days to predict the next one
X_train, y_train = make_windows(scaled_data, WINDOW, split)
X_test, y_test = make_windows(scaled_data, split, len(scaled_data))

# Define the RNN architecture
model = Sequential()
model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(WINDOW, 1)))
model.add(Dropout(0.2))
model.add(SimpleRNN(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model. With more than 30 epochs the loss does not improve much
model.fit(X_train, y_train, epochs=30, batch_size=32)

# Evaluate on the held-out test set (see test_model.py for the full report)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss (MSE on scaled prices): {test_loss:.6f}')

# Save the model and scaler for future use
model.save(str(MODEL_PATH))
with open(SCALER_PATH, 'wb') as file:
    pickle.dump(scaler, file)
