"""Validate the chosen rule as a complement to buy and hold, with data not used to choose it.

Strategy: hold the asset all the time and add --extra more (50% by default),
borrowed at the Treasury bill rate plus BORROW_SPREAD, while the rule is in a
trade. The rule buys when RSI(5) closes below 30 and the US 10-year yield fell
in the last 5 days, and sells 5 days later (see compare_exits.py).

It is compared with buying and holding the asset, and with holding it with the
same average extra exposure every day. The difference with the latter shows
whether the timing of the signal adds anything beyond investing more. For each
market, the 5-day return of the trades is also compared with that of any 5
days.

The rule was chosen with SPY from 2001 onwards, so the data used here is:
- SPY from 1993 to 2000,
- other markets through ETFs listed in the US (adjusted for dividends), over
  their whole history. They are other markets, but their history overlaps the
  period used to choose the rule and the 10-year yield filter is the same for
  all, so they are not fully independent tests.

The results are plotted in assets/overlay_validation.png.
"""
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np

from backtest import (ASSETS_DIR, TRADING_DAYS, Exit, add_indicators, download_markets,
                      period_slice, simulate)
from compare_exits import ENTRY, FILTER
from search_rules import entry_signals, filters

MARKETS = {   # name -> (ticker, period or None for the whole history)
    'S&P 500 (SPY), 1993-2000': ('SPY', ('1993-01-01', '2000-12-31')),
    'Nasdaq-100 (QQQ)': ('QQQ', None),
    'Russell 2000 (IWM)': ('IWM', None),
    'Germany (EWG)': ('EWG', None),
    'Japan (EWJ)': ('EWJ', None),
    'United Kingdom (EWU)': ('EWU', None),
    'France (EWQ)': ('EWQ', None),
    'Hong Kong (EWH)': ('EWH', None),
    'Australia (EWA)': ('EWA', None),
    'Canada (EWC)': ('EWC', None),
    'Brazil (EWZ)': ('EWZ', None),
}
BORROW_SPREAD = 1.0   # % per year paid over the Treasury bill rate on the borrowed money
COST = 0.001          # per purchase or sale of the extra position (these ETFs are less liquid than SPY)
EXIT = Exit('sell after 5 days', days=5)
WARM_UP = 30          # trading days that the indicators need before the first decision
PLOT_PATH = ASSETS_DIR / 'overlay_validation.png'


def cagr(returns):
    return np.prod(1 + returns) ** (TRADING_DAYS / len(returns)) - 1


def max_drawdown(returns):
    equity = np.cumprod(1 + returns)
    return (equity / np.maximum.accumulate(equity) - 1).min()


def evaluate_market(m, period, extra):
    days = period_slice(m.index, period)
    entry = (entry_signals(m)[ENTRY] & filters(m)[FILTER]).values
    _, position, trades = simulate(m, entry, EXIT)

    o = m['open'].values
    asset = np.append(o[1:] / o[:-1] - 1, 0.0)[days]
    borrow = (np.nan_to_num(np.append(np.nan, m['cash_rate'].values[:-1])) + BORROW_SPREAD)[days] / 100 / TRADING_DAYS
    signal_extra = extra * position[days]
    with_signal = asset + signal_extra * (asset - borrow) - COST * np.abs(np.diff(signal_extra, prepend=0.0))
    same_every_day = asset + signal_extra.mean() * (asset - borrow)

    trade_returns = np.array([o[end] / o[start] - 1 for start, end in trades if days[0] <= start <= days[-1]])
    any_5_days = o[days[0] + 5:days[-1] + 1] / o[days[0]:days[-1] - 4] - 1
    difference = trade_returns.mean() - any_5_days.mean()
    t = difference / math.sqrt(trade_returns.var(ddof=1) / len(trade_returns))
    return {
        'start': m.index[days[0]].year, 'end': m.index[days[-1]].year,
        'hold': cagr(asset), 'signal': cagr(with_signal), 'same': cagr(same_every_day),
        'hold_dd': max_drawdown(asset), 'signal_dd': max_drawdown(with_signal),
        'exposure': position[days].mean(), 'trades': len(trade_returns),
        'trade_mean': trade_returns.mean(), 'any_5_days': any_5_days.mean(),
        'p_value': math.erfc(abs(t) / math.sqrt(2)),
        'excess_trades': trade_returns - any_5_days.mean(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--extra', type=float, default=0.5,
                        help='extra exposure while the rule is in a trade (default: 0.5, i.e. 50%%)')
    args = parser.parse_args()

    markets = download_markets([ticker for ticker, _ in MARKETS.values()])
    print(f'Hold the asset and add {args.extra:.0%} more while the rule is in a trade '
          f'(buy when {ENTRY} and {FILTER}; sell after 5 days).')
    print(f'Borrowing at the Treasury bill rate + {BORROW_SPREAD:.0f}%; '
          f'{COST:.2%} per purchase or sale of the extra position.\n')
    print(f'{"":<27}{"years":>10}  {"annual return (CAGR)":^29}  {"timing":>7}  {"max drawdown":^15}  '
          f'{"trades":>6}  {"5-day return":^17}{"p":>7}')
    print(f'{"":<27}{"":>10}  {"hold":>9}{"+ signal":>10}{"+ same":>10}  {"gain":>7}  '
          f'{"hold":>7}{"+ signal":>9}  {"":>6}  {"trades":>8}{"any":>9}{"":>7}')

    results = {}
    for name, (ticker, period) in MARKETS.items():
        m = add_indicators(markets[ticker])
        r = evaluate_market(m, period or (m.index[WARM_UP], None), args.extra)
        results[name] = r
        print(f'{name:<27}{r["start"]:>5}-{r["end"]:<4}  {r["hold"]:9.2%}{r["signal"]:10.2%}{r["same"]:10.2%}  '
              f'{r["signal"] - r["same"]:+7.2%}  {r["hold_dd"]:7.1%}{r["signal_dd"]:9.1%}  {r["trades"]:6d}  '
              f'{r["trade_mean"]:+8.2%}{r["any_5_days"]:+9.2%}{r["p_value"]:7.3f}')

    gains = np.array([r['signal'] - r['same'] for r in results.values()])
    excess = np.concatenate([r['excess_trades'] for r in results.values()])
    t = excess.mean() / math.sqrt(excess.var(ddof=1) / len(excess))
    print(f'\nThe timing of the signal added return in {np.sum(gains > 0)} of {len(gains)} markets '
          f'(average {gains.mean():+.2%} per year).')
    print(f'All markets together: {len(excess)} trades, {excess.mean():+.2%} over any 5 days '
          f'(p-value {math.erfc(abs(t) / math.sqrt(2)):.3f}; markets move together, so this '
          'overstates the evidence).')
    print('\n+ signal: hold and add the extra while the rule is in a trade. + same: hold with the '
          'same average extra every day. timing gain: + signal minus + same. 5-day return: '
          'average of the trades and of any 5 days. p: p-value of their difference.')
    plot(results, args.extra)
    print(f'Plot saved to {PLOT_PATH}')


def plot(results, extra):
    names = list(results)[::-1]
    gains = [(results[n]['signal'] - results[n]['same']) * 100 for n in names]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names, gains, color='#257de0', height=0.6)
    ax.axvline(0, color='#c2c2c2', linewidth=1)
    ax.set_xlim(min(min(gains), 0) - 0.5, max(max(gains), 0) + 0.4)   # room for the labels
    for i, gain in enumerate(gains):
        ax.text(gain + (0.03 if gain >= 0 else -0.03), i, f'{gain:+.2f}', va='center',
                ha='left' if gain >= 0 else 'right', fontsize=9, color='#191919')
    ax.set_xlabel('Annual return added by the timing of the signal (percentage points)')
    ax.set_title(f'Adding {extra:.0%} while the rule is in a trade vs. the same extra every day',
                 loc='left', fontsize=11)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.tight_layout()
    fig.savefig(PLOT_PATH)


if __name__ == '__main__':
    main()
