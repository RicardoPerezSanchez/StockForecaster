"""Settings and helpers shared by train_model.py, test_model.py and walk_forward.py.

All the scripts must use the same data, features, window and train/test split,
so everything that defines them lives here.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

START_DATE = '2000-01-01'
END_DATE = '2023-01-01'

# Variable to predict and groups of input variables: name -> (ticker, kind).
# How each variable enters the model:
# - 'price': daily log return.
# - 'rate': daily change in percentage points (interest rates can be zero or
#   negative).
# - 'level': logarithm of the value, so the model sees how high it is (the
#   VIX tends to return to its usual levels).
# - 'ratio': logarithm of the first ticker divided by the second one.
TARGET = 'S&P 500'
TARGET_TICKER = '^GSPC'
FEATURE_GROUPS = {
    'markets': {
        'Nikkei 225': ('^N225', 'price'),
        'Hang Seng': ('^HSI', 'price'),
        'Shanghai Composite': ('000001.SS', 'price'),
        'FTSE 100': ('^FTSE', 'price'),
        'DAX': ('^GDAXI', 'price'),
        'Gold': ('GC=F', 'price'),
        'Crude oil (WTI)': ('CL=F', 'price'),
        'US 10-year yield': ('^TNX', 'rate'),
        'US 3-month yield': ('^IRX', 'rate'),
    },
    'vix': {
        'VIX': ('^VIX', 'level'),
        'SKEW': ('^SKEW', 'level'),
    },
    # VIX above its 3-month version means short-term fear. The 3-month VIX
    # starts in July 2006, so this group shortens the dataset.
    'vix_term': {
        'VIX / VIX 3-month': (('^VIX', '^VIX3M'), 'ratio'),
    },
}
MODEL_GROUPS = ('markets', 'vix')   # Groups used by train_model.py and test_model.py

WINDOW = 60              # Number of past days used as input for each prediction
TEST_SIZE = 0.2          # Fraction of the most recent days kept for testing
VALIDATION_SPLIT = 0.1   # Most recent fraction of the training data, for early stopping
SEED = 42

# Paths are resolved from this file, so the scripts work from any directory
ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / 'assets'
MODELS_DIR = ASSETS_DIR / 'models'
MODEL_PATH = MODELS_DIR / 'sp500_lstm_model.h5'
NORMALIZATION_PATH = MODELS_DIR / 'normalization.json'
PLOT_PATH = ASSETS_DIR / 'sp500_prediction.png'


def feature_specs(groups):
    """The S&P 500 plus the variables of the given groups: name -> (ticker, kind)."""
    specs = {TARGET: (TARGET_TICKER, 'price')}
    for group in groups:
        specs.update(FEATURE_GROUPS[group])
    return specs


def download_prices(groups=MODEL_GROUPS, start=START_DATE):
    """Download the closing values of the tickers used by the given groups.

    Returns a DataFrame with one column per ticker, indexed by the S&P 500
    trading days. Other markets have other holidays: each day takes the last
    known value of every ticker, so no future value is ever used. Days before
    every ticker has data are dropped (gold and crude start in August 2000).
    """
    specs = feature_specs(groups)
    tickers = []
    for ticker, _ in specs.values():
        tickers += [t for t in (ticker if isinstance(ticker, tuple) else (ticker,)) if t not in tickers]
    data = yf.download(tickers, start=start, end=END_DATE, progress=False)
    # yf.download does not raise when a download fails: it leaves the data empty
    close = data['Close'] if not data.empty else pd.DataFrame()
    missing = [t for t in tickers if t not in close or close[t].isna().all()]
    if missing:
        raise RuntimeError(
            f'No data downloaded for {", ".join(missing)}. Check the internet '
            'connection, or upgrade yfinance if Yahoo Finance changed its API.'
        )
    close = close[tickers].sort_index()
    for ticker, kind in specs.values():
        if kind == 'price':
            # Discard non-positive prices (WTI crude closed at -37.63 on 2020-04-20)
            close[ticker] = close[ticker].where(close[ticker] > 0)
    trading_days = close[TARGET_TICKER].dropna().index
    return close.ffill().loc[trading_days].dropna()


def load_dataset(groups=MODEL_GROUPS, start=START_DATE):
    """Download the data and compute the inputs of the model.

    Returns (features, close, previous_close). Row d of features holds the
    values known at the close of day d; close and previous_close are the
    S&P 500 closes on day d and on day d-1.
    """
    prices = download_prices(groups, start)
    features = pd.DataFrame(index=prices.index)
    for name, (ticker, kind) in feature_specs(groups).items():
        if kind == 'price':
            features[name] = np.log(prices[ticker]).diff()
        elif kind == 'rate':
            features[name] = prices[ticker].diff()
        elif kind == 'level':
            features[name] = np.log(prices[ticker])
        elif kind == 'ratio':
            first, second = ticker
            features[name] = np.log(prices[first] / prices[second])
        else:
            raise ValueError(f'Unknown kind of variable: {kind}')
    features = features.iloc[1:]
    close = prices[TARGET_TICKER].iloc[1:]
    previous_close = prices[TARGET_TICKER].shift(1).iloc[1:]
    return features, close, previous_close


def split_index(n_days):
    """Index of the first day in the test set.

    The split is chronological: the model is trained on the oldest days and
    tested on the most recent TEST_SIZE fraction.
    """
    split = int(n_days * (1 - TEST_SIZE))
    if split <= WINDOW or split >= n_days:
        raise ValueError(
            f'Not enough data ({n_days} days) for a {WINDOW}-day window '
            f'and a {TEST_SIZE:.0%} test set.'
        )
    return split


class Normalizer:
    """Standardize each feature with the mean and deviation of the training data."""

    def __init__(self, columns, mean, std):
        self.columns = list(columns)
        self.mean = np.asarray(mean, dtype='float64')
        self.std = np.asarray(std, dtype='float64')
        self.target_index = self.columns.index(TARGET)

    @classmethod
    def fit(cls, features):
        return cls(features.columns, features.mean().values, features.std().values)

    def transform(self, features):
        missing = [c for c in self.columns if c not in features]
        if missing:
            raise ValueError(f'Missing variables: {", ".join(missing)}. The model was trained '
                             'with other variables: run train_model.py again.')
        return (features[self.columns].values - self.mean) / self.std

    def to_return(self, z):
        """Convert standardized predictions of the target to log returns."""
        return z * self.std[self.target_index] + self.mean[self.target_index]

    def save(self, path):
        with open(path, 'w') as file:
            json.dump({'columns': self.columns, 'mean': self.mean.tolist(),
                       'std': self.std.tolist()}, file, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as file:
            data = json.load(file)
        return cls(data['columns'], data['mean'], data['std'])


def make_windows(z, target_index, start, end):
    """Build the samples whose targets are the days in [start, end).

    Each input holds the WINDOW previous days of every feature, all of them
    known at the close of the day before the target.
    """
    X = np.stack([z[i - WINDOW:i] for i in range(start, end)])
    y = z[start:end, target_index]
    return X, y


def train_model(X, y, seed, verbose=0):
    """Train a new model. Training stops when the validation loss stops improving."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = Sequential()
    model.add(LSTM(units=32, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    # validation_split takes the last samples before shuffling, so the model is
    # validated on the most recent part of the training period
    history = model.fit(X, y, epochs=100, batch_size=32, validation_split=VALIDATION_SPLIT,
                        callbacks=[early_stopping], verbose=verbose)
    return model, history


def predict_returns(model, normalizer, X):
    """Predicted log returns of the S&P 500."""
    return normalizer.to_return(model.predict(X, verbose=0)[:, 0])
