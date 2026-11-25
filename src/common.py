"""Settings and helpers shared by train_model.py and test_model.py.

Both scripts must use the same data, window size and train/test split, so
everything that defines them lives here.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

TICKER = '^GSPC'
START_DATE = '2000-01-01'
END_DATE = '2023-01-01'

WINDOW = 60       # Number of past days used as input for each prediction
TEST_SIZE = 0.2   # Fraction of the most recent days kept for testing
SEED = 42

# Paths are resolved from this file, so the scripts work from any directory
ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / 'assets'
MODELS_DIR = ASSETS_DIR / 'models'
MODEL_PATH = MODELS_DIR / 'sp500_rnn_model.h5'
SCALER_PATH = MODELS_DIR / 'scaler.pkl'
PLOT_PATH = ASSETS_DIR / 'sp500_prediction.png'


def download_prices():
    """Download the daily closing prices as a Series indexed by date.

    yf.download does not raise when the download fails: it returns an empty
    DataFrame, so check for that here and fail with a clear message.
    """
    data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
    if data.empty:
        raise RuntimeError(
            f'No data downloaded for {TICKER} ({START_DATE} to {END_DATE}). '
            'Check the internet connection, or upgrade yfinance if Yahoo '
            'Finance changed its API.'
        )
    close = data['Close']
    # Newer yfinance versions return one column per ticker
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


def split_index(n_prices):
    """Index of the first day in the test set.

    The split is chronological: the model is trained on the oldest days and
    tested on the most recent TEST_SIZE fraction.
    """
    split = int(n_prices * (1 - TEST_SIZE))
    if split <= WINDOW or split >= n_prices:
        raise ValueError(
            f'Not enough data ({n_prices} days) for a {WINDOW}-day window '
            f'and a {TEST_SIZE:.0%} test set.'
        )
    return split


def make_windows(scaled, start, end):
    """Build the samples whose targets are the days in [start, end).

    Each input holds the WINDOW previous values, which may belong to an
    earlier period: they are already known on the day of the prediction.
    """
    X = np.array([scaled[i - WINDOW:i, 0] for i in range(start, end)])
    y = scaled[start:end, 0]
    return X.reshape(-1, WINDOW, 1), y
