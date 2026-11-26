"""Search the long-only trading rules for SPY that earn the most money.

A rule has three parts: an entry signal, a filter that must also be true to
buy, and an exit. Every combination is simulated (see backtest.py) and:

1. ranked by its annual return (CAGR) in the training period,
2. the best TOP_N of the training period are compared in the validation
   period, and the best one there is chosen,
3. the chosen rule is evaluated in the test period, which no step above has
   used, next to buying and holding SPY and to keeping the money in cash.

The chosen rule is saved to assets/best_rule.json and its equity curve to
assets/rules_equity.png.
"""
import json
from dataclasses import asdict
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import pandas as pd

from backtest import (ASSETS_DIR, COST, MAX_HOLD, PERIODS, Exit, add_indicators,
                      buy_and_hold, cash_only, download_market_data, metrics,
                      period_slice, simulate)

TOP_N = 20
MIN_TRADES = 30   # minimum trades in the training period, to avoid flukes
RESULTS_PATH = ASSETS_DIR / 'best_rule.json'
PLOT_PATH = ASSETS_DIR / 'rules_equity.png'


def entry_signals(m):
    """Signals known at the close of a day; the purchase happens at the next open."""
    c = m['close']
    signals = {}
    for n, threshold in product((2, 3, 5), (10, 20, 30)):
        signals[f'RSI({n}) below {threshold}'] = m[f'rsi_{n}'] < threshold
    down = c < c.shift(1)
    for k in (2, 3, 4):
        signals[f'{k} down closes in a row'] = down.rolling(k).sum() == k
    for drop in (2, 3, 5):
        signals[f'close {drop}% below the 5-day high'] = c / c.rolling(5).max() - 1 <= -drop / 100
    for k in (1.5, 2):
        signals[f'close below the Bollinger band (20 days, {k} sd)'] = c < m['sma_20'] - k * m['std_20']
    for n in (5, 10, 20, 50):
        signals[f'close above SMA({n})'] = c > m[f'sma_{n}']
    for n in (5, 10, 20):
        signals[f'{n}-day closing high'] = c >= c.rolling(n).max()
    signals['VIX 20% above its 10-day average'] = m['vix'] > 1.2 * m['vix_mean_10']
    signals['VIX above the 3-month VIX (data since 2006)'] = m['vix_term'] > 1
    signals['turn of the month (buy on the last trading day)'] = m['before_last_day_of_month']
    signals['every day'] = pd.Series(True, index=m.index)
    return signals


def filters(m):
    c = m['close']
    high_volatility = m['volatility_20'] > m['volatility_median']
    low_volatility = m['volatility_20'] < m['volatility_median']
    return {
        'no filter': pd.Series(True, index=m.index),
        'above SMA(200)': c > m['sma_200'],
        'below SMA(200)': c < m['sma_200'],
        'high volatility': high_volatility,
        'low volatility': low_volatility,
        'world indices fell in 5 days': m['world_5d'] < 0,
        'world indices rose in 5 days': m['world_5d'] > 0,
        '10-year yield fell in 5 days': m['yield_10y_5d'] < 0,
        '10-year yield rose in 5 days': m['yield_10y_5d'] > 0,
        'gold rose in 20 days': m['gold_20d'] > 0,
        'gold fell in 20 days': m['gold_20d'] < 0,
        'crude rose in 20 days': m['crude_20d'] > 0,
        'crude fell in 20 days': m['crude_20d'] < 0,
        'VIX above its 1-year median': m['vix'] > m['vix_median'],
        'VIX below its 1-year median': m['vix'] < m['vix_median'],
        'VIX above 30': m['vix'] > 30,
        'VIX rose in 5 days': m['vix_5d'] > 0,
        'VIX fell in 5 days': m['vix_5d'] < 0,
        # Before July 2006 there is no 3-month VIX, so these filters never pass
        'VIX above the 3-month VIX (data since 2006)': m['vix_term'] > 1,
        'VIX below the 3-month VIX (data since 2006)': m['vix_term'] < 1,
        'SKEW above its 1-year median': m['skew'] > m['skew_median'],
        'SKEW below its 1-year median': m['skew'] < m['skew_median'],
    }


EXITS = [Exit(f'sell after {days} day{"s" if days > 1 else ""}', days) for days in range(1, MAX_HOLD + 1)] + [
    Exit('sell when RSI(2) is above 50', signal='rsi2>50'),
    Exit('sell when RSI(2) is above 70', signal='rsi2>70'),
    Exit('sell on a close above SMA(5)', signal='above_sma5'),
    Exit('sell after the first up close', signal='up_close'),
    Exit('sell on the first close above the purchase price', signal='profit'),
]


def describe(rule):
    entry, filter_name, exit_rule = rule
    text = f'Buy when {entry}' + ('' if filter_name == 'no filter' else f' and {filter_name}')
    if exit_rule.signal:
        text += f'; {exit_rule.name}, or after {exit_rule.days} days at most'
    else:
        text += f'; {exit_rule.name}'
    return text


def fmt(row):
    return (f'CAGR {row["cagr"]:7.2%}  total {row["total_return"]:8.1%}  max drawdown '
            f'{row["max_drawdown"]:7.1%}  Sharpe {row["sharpe"]:5.2f}  in market '
            f'{row["exposure"]:4.0%}  trades {row["trades"]:4d}  won {row["win_rate"]:4.0%}')


def main():
    m = add_indicators(download_market_data())
    entries, conditions = entry_signals(m), filters(m)
    rules = list(product(entries, conditions, EXITS))
    print(f'Data: {m.index[0].date()} to {m.index[-1].date()}. '
          f'Periods: ' + ', '.join(f'{p} {s} to {e or m.index[-1].date()}' for p, (s, e) in PERIODS.items()))
    print(f'Simulating {len(rules)} rules (cost {COST:.2%} per purchase or sale, '
          f'at most {MAX_HOLD} days per trade)...', flush=True)

    results = []
    for rule in rules:
        entry, filter_name, exit_rule = rule
        sim = simulate(m, (entries[entry] & conditions[filter_name]).values, exit_rule)
        train = metrics(m, *sim, 'train')
        if train['trades'] >= MIN_TRADES:
            results.append((rule, sim, train))

    # 1. Rank by return in the training period
    results.sort(key=lambda r: r[2]['cagr'], reverse=True)
    top = [(rule, sim, train, metrics(m, *sim, 'validation')) for rule, sim, train in results[:TOP_N]]
    # 2. Choose the best of them in the validation period
    best_rule, best_sim, _, _ = max(top, key=lambda r: r[3]['cagr'])

    benchmarks = {'Buy and hold SPY': buy_and_hold(m), 'Cash (Treasury bills)': cash_only(m),
                  'Buy every day, hold 5 days': simulate(m, entries['every day'].values, EXITS[MAX_HOLD - 1])}

    print(f'\nBest {TOP_N} rules in the training period (CAGR in train / validation / test):')
    for i, (rule, sim, train, validation) in enumerate(top, 1):
        test = metrics(m, *sim, 'test')
        mark = '  <- chosen' if rule == best_rule else ''
        print(f'{i:2d}. {train["cagr"]:6.2%} / {validation["cagr"]:6.2%} / {test["cagr"]:6.2%}  {describe(rule)}{mark}')

    report = {'rule': describe(best_rule), 'entry': best_rule[0], 'filter': best_rule[1],
              'exit': asdict(best_rule[2]), 'cost': COST, 'periods': {}}
    print(f'\nChosen rule: {describe(best_rule)}')
    for period in PERIODS:
        print(f'\n{period.capitalize()}')
        chosen = metrics(m, *best_sim, period)
        report['periods'][period] = {k: float(v) for k, v in chosen.items()}
        print(f'  {"Chosen rule":<28}{fmt(chosen)}')
        for name, sim in benchmarks.items():
            print(f'  {name:<28}{fmt(metrics(m, *sim, period))}')

    # The same rule with double costs, to check that it does not depend on cheap trading
    entry, filter_name, exit_rule = best_rule
    expensive = simulate(m, (entries[entry] & conditions[filter_name]).values, exit_rule, cost=2 * COST)
    print(f'\nChosen rule with costs of {2 * COST:.2%} per purchase or sale:')
    for period in PERIODS:
        print(f'  {period:<12}{fmt(metrics(m, *expensive, period, cost=2 * COST))}')

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w') as file:
        json.dump(report, file, indent=2)
    plot_equity(m, best_sim, benchmarks['Buy and hold SPY'], benchmarks['Cash (Treasury bills)'], describe(best_rule))
    print(f'\nRule saved to {RESULTS_PATH} and plot to {PLOT_PATH}')


def plot_equity(m, rule_sim, hold_sim, cash_sim, title):
    start = period_slice(m.index, 'train')[0]
    dates = m.index[start:]
    plt.figure(figsize=(14, 6))
    for (returns, _, _), label, color, width in ((hold_sim, 'Buy and hold SPY', '#f47230', 1.5),
                                                 (cash_sim, 'Cash (Treasury bills)', '#9c9c9c', 1.5),
                                                 (rule_sim, 'Chosen rule', '#257de0', 2)):
        plt.plot(dates, np.cumprod(1 + returns[start:]), color=color, linewidth=width, label=label)
    for period, shade in (('validation', '#f2f2f2'), ('test', '#e6e6e6')):
        days = period_slice(m.index, period)
        plt.axvspan(m.index[days[0]], m.index[days[-1]], color=shade, zorder=0)
        plt.text(m.index[days[0]], 1.02, f' {period}', transform=plt.gca().get_xaxis_transform(), color='#666666')
    plt.yscale('log')
    plain = FuncFormatter(lambda value, _: f'{value:g}')
    plt.gca().yaxis.set_major_formatter(plain)
    plt.gca().yaxis.set_minor_formatter(plain)
    plt.ylabel('Value of 1 dollar invested (log scale)')
    plt.title(title, fontsize=10, loc='left')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)


if __name__ == '__main__':
    main()
