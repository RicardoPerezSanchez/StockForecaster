# StockForecaster

LSTM model to predict the next day's close of the S&P 500. Besides the S&P 500 itself, the model uses US interest rates, gold, crude oil and five major world indices.

## Installation

Requires Python 3.8 to 3.11 (TensorFlow 2.13 is not available for newer versions).

```
pip install -r requirements.txt
```

## How it works

- **Data**: daily closes from Yahoo Finance, from 2000 to 2022. The inputs are the S&P 500, the Nikkei 225, the Hang Seng, the Shanghai Composite, the FTSE 100, the DAX, gold and WTI crude futures, and the US 10-year and 3-month Treasury yields (the list is in `src/common.py`). Gold and crude data start in August 2000, so the dataset starts there.
- **Features**: daily log returns of the prices and daily changes of the interest rates, standardized with the mean and deviation of the training period. Other markets have other holidays: each S&P 500 trading day takes the last known value of every variable.
- **Target**: the S&P 500 log return of the next day, predicted from the previous 60 days of every variable. The predicted price is the previous close moved by that return.
- **Model**: one LSTM layer of 32 units with dropout. Training stops when the loss on the most recent 10% of the training data stops improving.
- **Split**: chronological. The model is trained with the oldest 80% of the days and tested on the most recent 20%.

## Usage

The scripts can be run from any directory. The trained model, the normalization and the plot are stored in `assets`, which is created automatically at the same level as `src`.

1. `python src/train_model.py` trains the model and saves it.
2. `python src/test_model.py` evaluates the saved model on the test period and saves the plot to `assets/sp500_prediction.png`.
3. `python src/walk_forward.py` evaluates the approach more rigorously: before each year of the test period it retrains the model from scratch with all the previous data, with 3 different seeds. Use `--sp500-only` to train with the S&P 500 alone and measure what the other variables add, and `--help` for the other options. It takes several minutes.

## Evaluation

Each prediction uses the real data up to the day before, so the model is compared with the naive forecast that the next day closes at the previous day's price:

- RMSE, MAE and MAPE of the predicted prices, for the model and for the naive forecast.
- A Diebold-Mariano test, which tells whether the difference between both is statistically significant.
- The percentage of days on which the model gets the direction right, next to the percentage of days the S&P 500 rises (the accuracy of always predicting a rise).

The model is only useful if it is significantly better than these baselines.
