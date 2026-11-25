from common import (MODEL_PATH, MODELS_DIR, NORMALIZATION_PATH, SEED, WINDOW,
                    Normalizer, load_dataset, make_windows, split_index, train_model)

# Create the output folder before training, so saving cannot fail at the end
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Download the S&P 500 and the other variables from Yahoo Finance
features, close, previous_close = load_dataset()

# Chronological split: the most recent days are kept for testing
split = split_index(len(features))

# Standardize with the training period only, so no information from the test
# period leaks into the model
normalizer = Normalizer.fit(features.iloc[:split])
z = normalizer.transform(features)

# Each sample uses the previous 60 days of every variable to predict the
# S&P 500 return of the next day
X_train, y_train = make_windows(z, normalizer.target_index, WINDOW, split)

model, history = train_model(X_train, y_train, SEED, verbose=2)
print(f'Training stopped after {len(history.history["loss"])} epochs '
      f'(best validation loss: {min(history.history["val_loss"]):.4f})')

# Save the model and the normalization for future use
model.save(str(MODEL_PATH))
normalizer.save(NORMALIZATION_PATH)
print(f'Model saved to {MODELS_DIR}. Run test_model.py to evaluate it.')
