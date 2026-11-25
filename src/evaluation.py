"""Compare the model with simple baselines.

Each prediction is for the next day and uses the real prices up to the day
before, so the model is compared with the naive forecast (the next day closes
at the previous day's price) and, for the direction, with always predicting a
rise.
"""
import math

import numpy as np


def diebold_mariano(errors_model, errors_baseline):
    """Diebold-Mariano test on the squared errors of two forecasts.

    Returns (statistic, two-sided p-value). A positive statistic means that the
    model has larger errors than the baseline. The variance uses Newey-West
    weights, because large errors come in clusters (volatile periods).
    """
    d = errors_model ** 2 - errors_baseline ** 2
    n = len(d)
    lags = int(n ** (1 / 3))
    centered = d - d.mean()
    variance = centered @ centered / n
    for k in range(1, lags + 1):
        variance += 2 * (1 - k / (lags + 1)) * (centered[k:] @ centered[:-k]) / n
    if variance <= 0:
        return float('nan'), float('nan')
    statistic = d.mean() / math.sqrt(variance / n)
    return statistic, math.erfc(abs(statistic) / math.sqrt(2))


def price_errors(real, predicted):
    errors = predicted - real
    return {
        'RMSE': np.sqrt(np.mean(errors ** 2)),
        'MAE': np.mean(np.abs(errors)),
        'MAPE': np.mean(np.abs(errors / real)) * 100,
    }


def evaluate(real, previous_close, predicted):
    """Errors of the model and of the naive forecast, and direction accuracy."""
    statistic, p_value = diebold_mariano(predicted - real, previous_close - real)
    rises = real > previous_close
    return {
        'model': price_errors(real, predicted),
        'naive': price_errors(real, previous_close),
        'dm_statistic': statistic,
        'dm_p_value': p_value,
        'direction_accuracy': np.mean((predicted > previous_close) == rises) * 100,
        'rise_rate': np.mean(rises) * 100,
    }


def print_report(results, title=None):
    if title:
        print(f'\n{title}')
    print(f'{"":<28}{"RMSE":>9}{"MAE":>9}{"MAPE":>9}')
    for label, key in (('Model', 'model'), ('Naive (previous close)', 'naive')):
        e = results[key]
        print(f'{label:<28}{e["RMSE"]:9.2f}{e["MAE"]:9.2f}{e["MAPE"]:8.3f}%')

    p_value = results['dm_p_value']
    if p_value < 0.05:
        verdict = 'better' if results['dm_statistic'] < 0 else 'worse'
        verdict = f'the model is significantly {verdict} than the naive forecast'
    else:
        verdict = 'no significant difference with the naive forecast'
    print(f'Diebold-Mariano test: statistic {results["dm_statistic"]:.2f}, '
          f'p-value {p_value:.3f} ({verdict})')
    print(f'Direction: the model is right on {results["direction_accuracy"]:.1f}% of '
          f'the days; always predicting a rise is right on {results["rise_rate"]:.1f}%')
