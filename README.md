# StockForecaster

Simple RNN model to predict stock prices. For demo purposes the training and benchmarking is done with the prices of the SP500.

## Installation

Requires Python 3.8 to 3.11 (TensorFlow 2.13 is not available for newer versions).

```
pip install -r requirements.txt
```

## Usage

The scripts can be run from any directory. The trained model and the scaler are stored in `assets/models`, which is created automatically at the same level as `src`.

1. `python src/train_model.py` trains the model with the oldest 80% of the days and saves it.
2. `python src/test_model.py` evaluates the saved model on the most recent 20% of the days, which the model has not seen during training. It prints the RMSE, MAE and MAPE of the model next to those of a baseline that predicts the previous day's close, and saves the plot to `assets/sp500_prediction.png`.

Each prediction is for the next day and uses the real prices of the previous 60 days, so the model is only useful if it beats the baseline.
