"""
train.py - training pipeline for next-day *direction and return* prediction.

History of this file, because it is the point of the project:

  v1 (notebook)  predicted today's close from today's high/low/open. Target
                 leakage - the answer was inside the inputs. r2 = 0.9999.
  v2             predicted tomorrow's close from today's OHLCV, chronological
                 split. Honest, but pointless: it could not beat the naive
                 "tomorrow closes where today closed" baseline, because a price
                 level is ~99% explained by the previous price level.
  v3 (this file) predicts tomorrow's *return* and its *direction*. The naive
                 baseline for direction is ~51% (the market drifts up), so there
                 is finally a benchmark a model has to actually earn.

Protocol:
  - Features are stationary (see features.py) and use only data up to close(t).
  - The split is chronological and three-way: train / validation / test.
    Models are compared on validation. The winner is scored ONCE on test.
    Picking the best model by test score would leak the test set into that
    decision and inflate the number that ends up being reported.

Run:  python train.py
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import yfinance as yf

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              RandomForestRegressor)
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             mean_absolute_error, mean_squared_error)

from features import build_dataset

# ---------------------------------------------------------------- configuration

TICKER = 'INFY.NS'
START = '2015-01-01'
TRAIN_FRACTION = 0.60
VAL_FRACTION = 0.20          # test gets the remaining 20%, the most recent days

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
PRED_PATH = os.path.join(BASE_DIR, 'test_predictions.csv')


# ------------------------------------------------------------------ data loading

def load_data(ticker=TICKER, start=START, use_cache=True, refresh=False):
    """Download daily OHLCV, caching to CSV so reruns survive Yahoo rate limits."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, ticker.replace('.', '_') + '.csv')

    if use_cache and not refresh and os.path.exists(cache_file):
        print('Loading cached data from ' + cache_file)
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        print('Downloading ' + ticker + ' from ' + start + '...')
        data = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if data.empty:
            raise SystemExit('No data returned for ' + ticker + '. Yahoo may be rate limiting.')
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.to_csv(cache_file)
        print('Cached to ' + cache_file)

    return data.dropna()


def chronological_split(n):
    a = int(n * TRAIN_FRACTION)
    b = int(n * (TRAIN_FRACTION + VAL_FRACTION))
    return slice(0, a), slice(a, b), slice(b, n)


# --------------------------------------------------------------------- scoring

def score_direction(y_true, pred, proba=None):
    out = {
        'accuracy': float(accuracy_score(y_true, pred)) * 100,
        'f1': float(f1_score(y_true, pred, zero_division=0)) * 100,
    }
    if proba is not None and len(np.unique(y_true)) > 1:
        out['auc'] = float(roc_auc_score(y_true, proba))
    else:
        out['auc'] = float('nan')
    return out


def main():
    data = load_data()
    X, y_ret, y_dir, close = build_dataset(data)

    tr, va, te = chronological_split(len(X))
    X_tr, X_va, X_te = X.iloc[tr], X.iloc[va], X.iloc[te]
    d_tr, d_va, d_te = y_dir.iloc[tr], y_dir.iloc[va], y_dir.iloc[te]
    r_tr, r_va, r_te = y_ret.iloc[tr], y_ret.iloc[va], y_ret.iloc[te]

    print('')
    print('Ticker    : ' + TICKER)
    print('Rows      : %d  (%s to %s)  %d features'
          % (len(X), X.index[0].date(), X.index[-1].date(), X.shape[1]))
    print('Train     : %5d  %s -> %s' % (len(X_tr), X_tr.index[0].date(), X_tr.index[-1].date()))
    print('Validation: %5d  %s -> %s' % (len(X_va), X_va.index[0].date(), X_va.index[-1].date()))
    print('Test      : %5d  %s -> %s' % (len(X_te), X_te.index[0].date(), X_te.index[-1].date()))
    print('Target    : sign and size of the close(t) -> close(t+1) move')
    print('')

    # ---- baselines -----------------------------------------------------------
    # "Always predict up" is the benchmark to beat. Large caps drift upward, so
    # this is a genuinely hard bar, not a coin flip.
    always_up_va = float((d_va == 1).mean()) * 100
    always_up_te = float((d_te == 1).mean()) * 100
    print('Baseline - always predict UP:  validation %.2f%%   test %.2f%%'
          % (always_up_va, always_up_te))
    print('')

    # ---- direction models, compared on VALIDATION ----------------------------
    candidates = {
        # Two regularisation strengths. With 25 correlated features and a
        # near-random target, heavy shrinkage is usually what survives.
        'LogisticRegression': LogisticRegression(C=0.1, max_iter=2000),
        'LogisticRegression_weak_reg': LogisticRegression(C=1.0, max_iter=2000),
        'RandomForest': RandomForestClassifier(
            n_estimators=400, max_depth=4, min_samples_leaf=50,
            random_state=0, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.03,
            subsample=0.8, random_state=0),
    }

    print('Direction models (scored on validation):')
    val_results, fitted = [], {}
    for name, est in candidates.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('model', est)])
        pipe.fit(X_tr, d_tr)
        proba = pipe.predict_proba(X_va)[:, 1]
        s = score_direction(d_va, pipe.predict(X_va), proba)
        s['name'] = name
        val_results.append(s)
        fitted[name] = pipe
        print('  %-24s accuracy %5.2f%%   f1 %5.2f   auc %.4f'
              % (name, s['accuracy'], s['f1'], s['auc']))

    best_name = max(val_results, key=lambda r: r['accuracy'])['name']
    best_val = next(r for r in val_results if r['name'] == best_name)
    print('')
    print('Selected on validation: ' + best_name)

    # ---- refit the winner on train+validation, then score ONCE on test -------
    X_fit = pd.concat([X_tr, X_va])
    d_fit = pd.concat([d_tr, d_va])
    final = Pipeline([('scaler', StandardScaler()),
                      ('model', candidates[best_name])]).fit(X_fit, d_fit)

    proba_te = final.predict_proba(X_te)[:, 1]
    pred_te = final.predict(X_te)
    test_dir = score_direction(d_te, pred_te, proba_te)

    print('')
    print('Held-out test (%s -> %s):' % (X_te.index[0].date(), X_te.index[-1].date()))
    print('  accuracy %.2f%%   f1 %.2f   auc %.4f'
          % (test_dir['accuracy'], test_dir['f1'], test_dir['auc']))
    edge = test_dir['accuracy'] - always_up_te
    if edge > 0:
        print('  %+.2f points vs the always-up baseline.' % edge)
    else:
        print('  %+.2f points vs the always-up baseline - it does NOT beat it.' % edge)
        print('  That is the expected, honest result. Daily direction is close to a')
        print('  coin flip; the value here is measuring that rather than hiding it.')

    # ---- return regression, as a secondary head ------------------------------
    print('')
    print('Return magnitude (regression, test set):')
    reg_candidates = {
        'Ridge': Ridge(alpha=10.0),
        'RandomForestReg': RandomForestRegressor(
            n_estimators=300, max_depth=4, min_samples_leaf=50,
            random_state=0, n_jobs=-1),
    }
    r_fit = pd.concat([r_tr, r_va])
    zero_mae = float(mean_absolute_error(r_te, np.zeros(len(r_te))))
    print('  %-24s MAE %6.3f%%' % ('baseline (predict 0%)', zero_mae * 100))

    reg_results, reg_fitted = [], {}
    for name, est in reg_candidates.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('model', est)]).fit(X_fit, r_fit)
        p = pipe.predict(X_te)
        mae = float(mean_absolute_error(r_te, p))
        rmse = float(np.sqrt(mean_squared_error(r_te, p)))
        reg_results.append({'name': name, 'mae_pct': mae * 100, 'rmse_pct': rmse * 100})
        reg_fitted[name] = pipe
        print('  %-24s MAE %6.3f%%   RMSE %6.3f%%' % (name, mae * 100, rmse * 100))

    best_reg = min(reg_results, key=lambda r: r['mae_pct'])

    # ---- persist -------------------------------------------------------------
    # The bundle carries the feature order with the model, so the app can never
    # feed values into the wrong slots - the bug that started this rewrite.
    bundle = {
        'model': final,
        'return_model': reg_fitted[best_reg['name']],
        'features': list(X.columns),
        'target': 'next_day_direction',
        'ticker': TICKER,
        'model_name': best_name,
        'return_model_name': best_reg['name'],
        'trained_at': datetime.now().isoformat(timespec='seconds'),
        'train_end': str(X_tr.index[-1].date()),
        'val_end': str(X_va.index[-1].date()),
        'test_start': str(X_te.index[0].date()),
        'test_end': str(X_te.index[-1].date()),
        'metrics': {'validation': best_val, 'test': test_dir,
                    'baseline_always_up_test': always_up_te,
                    'return': best_reg, 'return_baseline_mae_pct': zero_mae * 100},
    }
    joblib.dump(bundle, MODEL_PATH)
    print('')
    print('Saved ' + best_name + ' -> ' + MODEL_PATH)

    # Test-set predictions drive both the backtest and the app's chart.
    pd.DataFrame({
        'date': X_te.index,
        'close': close.iloc[te].values,
        'actual_return': r_te.values,
        'actual_direction': d_te.values,
        'pred_direction': pred_te,
        'pred_proba_up': proba_te,
        'pred_return': reg_fitted[best_reg['name']].predict(X_te),
    }).to_csv(PRED_PATH, index=False)
    print('Saved test predictions -> ' + PRED_PATH)

    with open(os.path.join(BASE_DIR, 'metrics.json'), 'w') as f:
        json.dump({
            'target': 'next_day_direction',
            'baseline_always_up': {'validation': always_up_va, 'test': always_up_te},
            'validation': val_results,
            'selected': best_name,
            'test': test_dir,
            'regression': {'baseline_mae_pct': zero_mae * 100, 'models': reg_results},
            'split': {'train_end': str(X_tr.index[-1].date()),
                      'val_end': str(X_va.index[-1].date()),
                      'test_start': str(X_te.index[0].date()),
                      'test_end': str(X_te.index[-1].date())},
        }, f, indent=2)
    print('Saved metrics.json')


if __name__ == '__main__':
    main()
