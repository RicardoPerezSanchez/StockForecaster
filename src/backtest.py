"""Data, indicators and simulation for long-only trading rules on SPY.

How a rule is simulated:
- Every decision uses the data known at the close of a day and is executed
  at the open of the next day.
- A trade buys SPY with all the capital and sells it after at most MAX_HOLD
  trading days. A new trade can only start the day after the previous one
  ends.
- Every purchase and every sale costs COST (commission, spread and slippage).
- While out of the market the money earns the 3-month Treasury bill rate.
- SPY prices are adjusted for dividends, so holding SPY includes them.
"""
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

TICKER = 'SPY'
CASH_TICKER = '^IRX'   # 3-month Treasury bill rate, in % per year
WORLD_INDICES = ['^N225', '^HSI', '000001.SS', '^FTSE', '^GDAXI']
SIGNAL_TICKERS = {'gold': 'GC=F', 'crude': 'CL=F', 'yield_10y': '^TNX'}

DATA_START = '1993-01-01'   # SPY starts trading in 1993; the early years warm up the indicators
PERIODS = {
    'train': ('2001-01-01', '2014-12-31'),        # used to search the rules
    'validation': ('2015-01-01', '2019-12-31'),   # used to choose among the best ones
    'test': ('2020-01-01', None),                 # only used to report the result
}
MAX_HOLD = 5        # maximum trading days in a trade
COST = 0.0005       # cost of each purchase or sale (0.05%)
TRADING_DAYS = 252

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / 'assets'


def download_market_data():
    """SPY open and close, cash rate and signal variables on SPY's trading days.

    Other markets have other holidays: each day takes the last known value of
    every variable. Only full days are used (the data ends yesterday).
    """
    tickers = [TICKER, CASH_TICKER, *WORLD_INDICES, *SIGNAL_TICKERS.values()]
    data = yf.download(tickers, start=DATA_START, end=date.today().isoformat(),
                       progress=False)
    # yf.download does not raise when a download fails: it leaves the data empty
    close = data['Close'] if not data.empty else pd.DataFrame()
    missing = [t for t in tickers if t not in close or close[t].isna().all()]
    if missing:
        raise RuntimeError(f'No data downloaded for {", ".join(missing)}.')

    close = close.sort_index()
    # A price must be positive (WTI crude closed at -37.63 on 2020-04-20)
    crude = SIGNAL_TICKERS['crude']
    close[crude] = close[crude].where(close[crude] > 0)
    trading_days = data['Open'][TICKER].dropna().index
    known = close.ffill().loc[trading_days]

    market = pd.DataFrame({
        'open': data['Open'][TICKER].loc[trading_days],
        'close': data['Close'][TICKER].loc[trading_days],
        'cash_rate': known[CASH_TICKER],
    })
    for name, ticker in SIGNAL_TICKERS.items():
        market[name] = known[ticker]
    for ticker in WORLD_INDICES:
        market[ticker] = known[ticker]
    return market


def rsi(close, n):
    """Relative Strength Index with Wilder's smoothing."""
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def add_indicators(market):
    """Indicators used by the rules. Each value only uses data up to that day."""
    m = market.copy()
    c = m['close']
    for n in (2, 3, 5):
        m[f'rsi_{n}'] = rsi(c, n)
    for n in (5, 10, 20, 50, 200):
        m[f'sma_{n}'] = c.rolling(n).mean()
    m['std_20'] = c.rolling(20).std()
    log_return = np.log(c).diff()
    m['volatility_20'] = log_return.rolling(20).std()
    m['volatility_median'] = m['volatility_20'].rolling(TRADING_DAYS).median()
    m['world_5d'] = np.log(m[WORLD_INDICES]).diff(5).mean(axis=1)
    m['yield_10y_5d'] = m['yield_10y'].diff(5)
    m['gold_20d'] = np.log(m['gold']).diff(20)
    m['crude_20d'] = np.log(m['crude']).diff(20)
    # The next trading day is the last one of its month (the calendar of
    # trading days is known in advance)
    month = pd.Series(m.index.month, index=m.index)
    m['before_last_day_of_month'] = month.shift(-1) != month.shift(-2)
    return m


@dataclass(frozen=True)
class Exit:
    """When to sell: after `days` days at most, or earlier if `signal` is true at a close."""
    name: str
    days: int = MAX_HOLD
    signal: str = None   # 'rsi2>50', 'rsi2>70', 'above_sma5', 'up_close', 'profit'


def simulate(m, entry, exit_rule, cost=COST):
    """Simulate a rule. Returns (daily returns, position, trades).

    entry: boolean array, true when the rule wants to buy at the next open.
    position[d] is 1 when SPY is held from the open of day d to the next open.
    trades: list of (first day held, day of the sale), as indices.
    """
    o = m['open'].values
    c = m['close'].values
    n = len(o)
    rsi2, sma5 = m['rsi_2'].values, m['sma_5'].values
    candidates = np.flatnonzero(np.asarray(entry, dtype=bool))

    position = np.zeros(n)
    trades = []
    next_decision = 0
    while True:
        k = np.searchsorted(candidates, next_decision)
        if k == len(candidates) or candidates[k] + 1 >= n - 1:
            break
        start = candidates[k] + 1   # bought at this open
        end = n - 1
        for d in range(start, min(start + exit_rule.days, n - 1)):   # closes while held
            if d == start + exit_rule.days - 1 or _exit_signal(exit_rule.signal, d, start, o, c, rsi2, sma5):
                end = d + 1   # sold at the next open
                break
        position[start:end] = 1
        trades.append((start, end))
        next_decision = end

    open_to_open = np.append(o[1:] / o[:-1] - 1, 0.0)
    cash = np.nan_to_num(np.append(np.nan, m['cash_rate'].values[:-1])) / 100 / TRADING_DAYS
    changes = np.abs(np.diff(position, prepend=0.0))
    returns = position * open_to_open + (1 - position) * cash - cost * changes
    return returns, position, trades


def _exit_signal(signal, d, start, o, c, rsi2, sma5):
    if signal is None:
        return False
    if signal == 'rsi2>50':
        return rsi2[d] > 50
    if signal == 'rsi2>70':
        return rsi2[d] > 70
    if signal == 'above_sma5':
        return c[d] > sma5[d]
    if signal == 'up_close':
        return c[d] > c[d - 1]
    if signal == 'profit':
        return c[d] > o[start]
    raise ValueError(f'Unknown exit signal: {signal}')


def period_slice(index, period):
    start, end = PERIODS[period]
    days = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end or index[-1]))
    return np.flatnonzero(days)


def metrics(m, returns, position, trades, period, cost=COST):
    """Performance of a simulated rule in one of the PERIODS."""
    days = period_slice(m.index, period)
    r = returns[days]
    cash = np.nan_to_num(np.append(np.nan, m['cash_rate'].values[:-1]))[days] / 100 / TRADING_DAYS
    equity = np.cumprod(1 + r)
    years = len(r) / TRADING_DAYS
    o = m['open'].values
    trade_returns = np.array([o[end] / o[start] * (1 - cost) ** 2 - 1
                              for start, end in trades if days[0] <= start <= days[-1]])
    return {
        'total_return': equity[-1] - 1,
        'cagr': equity[-1] ** (1 / years) - 1,
        'volatility': r.std() * np.sqrt(TRADING_DAYS),
        'sharpe': (r - cash).mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else 0.0,
        'max_drawdown': (equity / np.maximum.accumulate(equity) - 1).min(),
        'exposure': position[days].mean(),
        'trades': len(trade_returns),
        'win_rate': (trade_returns > 0).mean() if len(trade_returns) else 0.0,
        'avg_trade': trade_returns.mean() if len(trade_returns) else 0.0,
    }


def buy_and_hold(m):
    """Returns of holding SPY all the time (one purchase, no time limit)."""
    position = np.ones(len(m))
    o = m['open'].values
    returns = np.append(o[1:] / o[:-1] - 1, 0.0)
    return returns, position, []


def cash_only(m):
    """Returns of keeping the money in Treasury bills."""
    cash = np.nan_to_num(np.append(np.nan, m['cash_rate'].values[:-1])) / 100 / TRADING_DAYS
    return cash, np.zeros(len(m)), []
