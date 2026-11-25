"""Compare ways of selling for the rule chosen by search_rules.py.

The rule buys SPY when RSI(5) closes below 30 and the 10-year yield fell in
the last 5 days. The exits compared are:
- selling after 5 days (the rule chosen by search_rules.py),
- selling when RSI(5) closes above 70, the opposite signal, or after 5 days
  at most,
- selling when RSI(5) closes above 70, with no time limit.

They are compared with buying and holding SPY and with keeping the money in
cash. The entry was chosen with the data up to 2019, so only the test period
(2020 onwards) is a fair estimate. The equity curves are saved to
assets/exits_equity.png.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from backtest import (ASSETS_DIR, PERIODS, Exit, add_indicators, buy_and_hold,
                      cash_only, download_market_data, metrics, period_slice, simulate)
from search_rules import entry_signals, filters

ENTRY = 'RSI(5) below 30'
FILTER = '10-year yield fell in 5 days'
EXITS = {
    'Sell after 5 days': Exit('sell after 5 days', days=5),
    'Sell at RSI(5) > 70 or after 5 days': Exit('sell when RSI(5) is above 70', days=5, signal='rsi5>70'),
    'Sell at RSI(5) > 70, no time limit': Exit('sell when RSI(5) is above 70', days=None, signal='rsi5>70'),
}
FULL_PERIOD = (PERIODS['train'][0], None)
PLOT_PATH = ASSETS_DIR / 'exits_equity.png'
COLORS = {'Sell after 5 days': '#257de0', 'Buy and hold SPY': '#f47230',
          'Sell at RSI(5) > 70 or after 5 days': '#00b894',
          'Sell at RSI(5) > 70, no time limit': '#f2a600', 'Cash (Treasury bills)': '#9c9c9c'}


def main():
    m = add_indicators(download_market_data())
    entry = (entry_signals(m)[ENTRY] & filters(m)[FILTER]).values
    strategies = {name: simulate(m, entry, exit_rule) for name, exit_rule in EXITS.items()}
    strategies['Buy and hold SPY'] = buy_and_hold(m)
    strategies['Cash (Treasury bills)'] = cash_only(m)

    print(f'Buy when {ENTRY} and {FILTER}, at the next open.')
    periods = {f'{p} ({s[:4]}-{e[:4] if e else "today"})': p for p, (s, e) in PERIODS.items()}
    periods[f'whole period ({FULL_PERIOD[0][:4]}-today)'] = FULL_PERIOD
    for title, period in periods.items():
        print(f'\n{title.capitalize()}')
        print(f'  {"":<37}{"CAGR":>7}{"total":>9}{"max DD":>8}{"in mkt":>8}{"trades":>8}{"days":>6}{"won":>6}')
        for name, sim in strategies.items():
            r = metrics(m, *sim, period)
            trade_columns = (f'{r["trades"]:8d}{r["avg_days"]:6.1f}{r["win_rate"]:6.0%}'
                             if r['trades'] else f'{"-":>8}{"-":>6}{"-":>6}')
            print(f'  {name:<37}{r["cagr"]:7.2%}{r["total_return"]:9.0%}{r["max_drawdown"]:8.1%}'
                  f'{r["exposure"]:8.0%}{trade_columns}')
    print('\nCAGR: annual return. max DD: largest fall from a previous high. in mkt: share of days '
          'holding SPY. days: average days per trade. won: share of trades with a profit.')

    plot(m, strategies)
    print(f'Plot saved to {PLOT_PATH}')


def plot(m, strategies):
    start = period_slice(m.index, FULL_PERIOD)[0]
    dates = m.index[start:]
    fig, ax = plt.subplots(figsize=(14, 6.5))
    for name, (returns, _, _) in strategies.items():
        equity = np.cumprod(1 + returns[start:])
        width = 1.5 if name in ('Buy and hold SPY', 'Cash (Treasury bills)') else 2
        ax.plot(dates, equity, color=COLORS[name], linewidth=width, label=name)
        ax.annotate(f'{equity[-1]:.1f}', xy=(dates[-1], equity[-1]), xytext=(6, 0),
                    textcoords='offset points', va='center', color='#191919', fontsize=9)
    for period, shade in (('validation', '#f2f2f2'), ('test', '#e6e6e6')):
        days = period_slice(m.index, period)
        ax.axvspan(m.index[days[0]], m.index[days[-1]], color=shade, zorder=0)
        ax.text(m.index[days[0]], 1.01, f' {period}', transform=ax.get_xaxis_transform(), color='#666666')
    ax.set_yscale('log')
    plain = FuncFormatter(lambda value, _: f'{value:g}')
    ax.yaxis.set_major_formatter(plain)
    ax.yaxis.set_minor_formatter(plain)
    ax.set_ylabel('Value of 1 dollar invested (log scale)')
    ax.set_title(f'Buy when {ENTRY} and {FILTER}: ways of selling', fontsize=11, loc='left', pad=18)
    ax.legend(frameon=False, loc='upper left')
    ax.margins(x=0.04)
    fig.tight_layout()
    fig.savefig(PLOT_PATH)


if __name__ == '__main__':
    main()
