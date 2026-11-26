# StockForecaster

LSTM model to predict the next day's close of the S&P 500. Besides the S&P 500 itself, the model uses US interest rates, gold, crude oil and five major world indices.

## Installation

Requires Python 3.8 to 3.11 (TensorFlow 2.13 is not available for newer versions).

```
pip install -r requirements.txt
```

## How it works

- **Data**: daily closes from Yahoo Finance, from 2000 to 2022. Besides the S&P 500, the inputs come in groups, defined in `src/common.py`:
  - `markets`: the Nikkei 225, the Hang Seng, the Shanghai Composite, the FTSE 100, the DAX, gold and WTI crude futures, and the US 10-year and 3-month Treasury yields.
  - `vix`: the VIX (expected volatility of the S&P 500) and the SKEW index (demand for protection against large falls).
  - `vix_term`: the VIX divided by the 3-month VIX; above 1 it signals short-term fear. Its data starts in July 2006.

  The model uses only `markets` (`MODEL_GROUPS`): in `walk_forward.py`, adding `vix` or `vix_term` made the predictions slightly worse. Gold and crude data start in August 2000, so the dataset starts there.
- **Features**: daily log returns of the prices, daily changes of the interest rates and the logarithm of the VIX, the SKEW and the VIX ratio, standardized with the mean and deviation of the training period. Other markets have other holidays: each S&P 500 trading day takes the last known value of every variable.
- **Target**: the S&P 500 log return of the next day, predicted from the previous 60 days of every variable. The predicted price is the previous close moved by that return.
- **Model**: one LSTM layer of 32 units with dropout. Training stops when the loss on the most recent 10% of the training data stops improving.
- **Split**: chronological. The model is trained with the oldest 80% of the days and tested on the most recent 20%.

## Usage

The scripts can be run from any directory. The trained model, the normalization and the plot are stored in `assets`, which is created automatically at the same level as `src`.

1. `python src/train_model.py` trains the model and saves it.
2. `python src/test_model.py` evaluates the saved model on the test period and saves the plot to `assets/sp500_prediction.png`.
3. `python src/walk_forward.py` evaluates the approach more rigorously: before each year of the test period it retrains the model from scratch with all the previous data, with 3 different seeds. Use `--groups` to choose the groups of variables (for example `--groups markets` to leave out the VIX, or `--groups` alone for the S&P 500 only) and `--start` to compare groups over the same days, since some variables start later. `--help` lists all the options. It takes several minutes.

## Evaluation

Each prediction uses the real data up to the day before, so the model is compared with the naive forecast that the next day closes at the previous day's price:

- RMSE, MAE and MAPE of the predicted prices, for the model and for the naive forecast.
- A Diebold-Mariano test, which tells whether the difference between both is statistically significant.
- The percentage of days on which the model gets the direction right, next to the percentage of days the S&P 500 rises (the accuracy of always predicting a rise).

The model is only useful if it is significantly better than these baselines.

## Trading rules

`python src/search_rules.py` searches long-only trading rules for SPY (the S&P 500 ETF, dividends included) with trades of at most 5 trading days. It takes less than a minute.

- **Rules**: an entry signal (oversold RSI, down closes in a row, drops from the 5-day high, Bollinger bands, moving averages, new highs, VIX spikes, turn of the month), a filter that must also hold (200-day trend, volatility, world indices, 10-year yield, gold, crude, VIX, VIX term structure, SKEW) and an exit (after 1 to 5 days, or earlier on an RSI, moving average or price signal).
- **Simulation** (`src/backtest.py`): decisions use the data known at a close and are executed at the next open. Each purchase or sale costs 0.05%, and the money out of the market earns the 3-month Treasury bill rate.
- **Selection**: the rules are ranked by their annual return in 2001-2014, the best 20 are compared in 2015-2019 and the best one there is chosen. It is then evaluated from 2020 onwards, a period not used to choose it, next to buying and holding SPY and to keeping the money in cash. It is also checked with double costs.

The chosen rule is saved to `assets/best_rule.json` and its equity curve to `assets/rules_equity.png`.

`python src/compare_exits.py` compares ways of selling for the chosen rule (buy when RSI(5) closes below 30 and the 10-year yield fell in 5 days): after 5 days, when RSI(5) closes above 70 or after 5 days, and when RSI(5) closes above 70 with no time limit. It prints the results of each period next to buy and hold and cash, and saves the equity curves to `assets/exits_equity.png`. Past results do not guarantee future ones: the more rules are tried, the more likely the best one is good by chance, which is why only the test period gives a fair estimate.
