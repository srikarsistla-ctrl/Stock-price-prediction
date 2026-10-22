"""
train.py - corrected training pipeline for the stock close price predictor.

The original notebook had two problems that made its r2 score of 0.9999 meaningless:

  1. Target leakage. It predicted the SAME day's close using that day's high, low
     and open. Those are only known after the market shuts, so the answer was
     already contained in the inputs. Here the target is the NEXT day's close.

  2. Shuffled split. train_test_split shuffles by default, which scatters future
     days into the training set on time series data. Here the split is
     chronological: the model trains on the past and is tested on the future.

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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------- configuration

TICKER = 'INFY.NS'          # NSE listing. Use 'INFY' for the NYSE ADR.
START = '2015-01-01'
TEST_FRACTION = 0.2         # the most recent 20% of days become the test set

FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')


# ------------------------------------------------------------------ data loading

def load_data(ticker=TICKER, start=START, use_cache=True):
    """Download daily OHLCV, caching to CSV so reruns survive Yahoo rate limits."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f'{ticker.replace(".", "_")}.csv')

    if use_cache and os.path.exists(cache_file):
        print(f'Loading cached data from {cache_file}')
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        print(f'Downloading {ticker} from {start}...')
        data = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if data.empty:
            raise SystemExit(
                f'No data returned for {ticker}. Yahoo may be rate limiting you - '
                'wait a few minutes and try again.'
            )
        # Recent yfinance returns MultiIndex columns like ('Close', 'INFY.NS').
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.to_csv(cache_file)
        print(f'Cached to {cache_file}')

    return data.dropna()


def build_dataset(data):
    """Today's OHLCV predicts tomorrow's close."""
    df = data[FEATURES].copy()
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()                      # last row has no tomorrow, so drop it
    return df[FEATURES], df['Target'], df['Close']


# --------------------------------------------------------------------- evaluation

def evaluate(name, y_true, y_pred, close_today):
    """Report error metrics plus directional accuracy, which is what actually matters."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Did we at least get the direction of the move right? The naive baseline
    # predicts no change at all, so it has no directional opinion to score.
    actual_up = np.sign(np.asarray(y_true) - np.asarray(close_today))
    pred_up = np.sign(np.asarray(y_pred) - np.asarray(close_today))
    if np.allclose(np.asarray(y_pred), np.asarray(close_today)):
        direction = float('nan')
        shown = '   n/a'
    else:
        direction = float(np.mean(actual_up == pred_up) * 100)
        shown = f'{direction:5.1f}%'

    print(f'  {name:<22} MAE {mae:8.2f}   RMSE {rmse:8.2f}   '
          f'r2 {r2:7.4f}   direction {shown}')

    return {'name': name, 'mae': mae, 'rmse': rmse,
            'r2': r2, 'direction_pct': direction}


# --------------------------------------------------------------------------- main

def main():
    data = load_data()
    X, y, close = build_dataset(data)

    # Chronological split - train on the past, test on the future.
    split = int(len(X) * (1 - TEST_FRACTION))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    close_test = close.iloc[split:]

    print(f'\nTicker           : {TICKER}')
    print(f'Rows             : {len(X)}  ({X.index[0].date()} to {X.index[-1].date()})')
    print(f'Train            : {len(X_train)}  (up to {X_train.index[-1].date()})')
    print(f'Test             : {len(X_test)}  (from {X_test.index[0].date()})')
    print(f'Features         : {FEATURES}')
    print(f'Target           : next trading day close\n')

    print('Baseline - assume tomorrow closes exactly where today closed:')
    baseline = evaluate('naive (no model)', y_test, close_test, close_test)

    print('\nModels:')
    candidates = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1, max_iter=10000),
        'RandomForest': RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1),
    }

    results = []
    fitted = {}
    for name, estimator in candidates.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('model', estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append(evaluate(name, y_test, pred, close_test))
        fitted[name] = pipe

    best = min(results, key=lambda r: r['rmse'])

    print(f'\nBest model by RMSE: {best["name"]} (RMSE {best["rmse"]:.2f})')
    if best['rmse'] < baseline['rmse']:
        margin = (baseline['rmse'] - best['rmse']) / baseline['rmse'] * 100
        print(f'It beats the naive baseline by {margin:.2f}%.')
    else:
        print(f'It does NOT beat the naive baseline (RMSE {baseline["rmse"]:.2f}).')
        print('That is the normal, honest result for daily price prediction - the')
        print('previous close is a very hard benchmark to improve on.')

    # Save the pipeline together with the metadata the app needs. The original
    # model.pkl was a bare estimator with no record of its feature order, which is
    # exactly how the app ended up feeding it inputs in the wrong slots.
    bundle = {
        'model': fitted[best['name']],
        'features': FEATURES,
        'target': 'next_day_close',
        'ticker': TICKER,
        'model_name': best['name'],
        'trained_at': datetime.now().isoformat(timespec='seconds'),
        'train_end': str(X_train.index[-1].date()),
        'test_start': str(X_test.index[0].date()),
        'metrics': best,
        'baseline': baseline,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f'\nSaved {best["name"]} -> {MODEL_PATH}')

    with open(os.path.join(BASE_DIR, 'metrics.json'), 'w') as f:
        json.dump({'baseline': baseline, 'models': results, 'best': best['name']}, f, indent=2)
    print('Saved metrics.json')


if __name__ == '__main__':
    main()
