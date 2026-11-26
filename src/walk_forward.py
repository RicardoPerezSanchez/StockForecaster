"""Walk-forward evaluation over the test period.

The test period is split into blocks of --retrain-every days. Before each block
the model is retrained from scratch with all the data up to that block, once
per seed, so every prediction only uses information available at that time.
The results are reported for each seed and for the average of all of them.
"""
import argparse

import numpy as np

from common import (FEATURE_GROUPS, MODEL_GROUPS, SEED, START_DATE, WINDOW, Normalizer,
                    load_dataset, make_windows, predict_returns, split_index, train_model)
from evaluation import evaluate, print_report

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--seeds', type=int, default=3,
                    help='number of random seeds (default: 3)')
parser.add_argument('--retrain-every', type=int, default=252,
                    help='trading days between retrainings (default: 252, about a year)')
parser.add_argument('--groups', nargs='*', default=list(MODEL_GROUPS), choices=list(FEATURE_GROUPS),
                    help='groups of input variables besides the S&P 500 (default: %(default)s). '
                         'Pass --groups alone to use only the S&P 500')
parser.add_argument('--start', default=START_DATE,
                    help='first date of the data (default: %(default)s). Use the same date to '
                         'compare groups over the same days, since some variables start later')
args = parser.parse_args()

features, close, previous_close = load_dataset(args.groups, args.start)
n_days = len(features)
split = split_index(n_days)
seeds = [SEED + i for i in range(args.seeds)]
blocks = [(start, min(start + args.retrain_every, n_days))
          for start in range(split, n_days, args.retrain_every)]

print(f'Inputs: {", ".join(features.columns)}')
print(f'Test period: {close.index[split].date()} to {close.index[-1].date()} '
      f'({n_days - split} days), {len(blocks)} blocks x {len(seeds)} seeds')

predicted_returns = np.zeros((len(seeds), n_days - split))
for start, end in blocks:
    # Standardize with the data available before the block only
    normalizer = Normalizer.fit(features.iloc[:start])
    z = normalizer.transform(features)
    X_train, y_train = make_windows(z, normalizer.target_index, WINDOW, start)
    X_block, _ = make_windows(z, normalizer.target_index, start, end)
    for i, seed in enumerate(seeds):
        model, history = train_model(X_train, y_train, seed)
        predicted_returns[i, start - split:end - split] = predict_returns(model, normalizer, X_block)
        print(f'  Block {close.index[start].date()} to {close.index[end - 1].date()}, '
              f'seed {seed}: {len(history.history["loss"])} epochs', flush=True)

real = close.values[split:]
previous = previous_close.values[split:]
for i, seed in enumerate(seeds):
    print_report(evaluate(real, previous, previous * np.exp(predicted_returns[i])),
                 title=f'Seed {seed}')
print_report(evaluate(real, previous, previous * np.exp(predicted_returns.mean(axis=0))),
             title='Average of all seeds')
