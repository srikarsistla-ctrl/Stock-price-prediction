"""
utils.py - live data, prediction, and the prediction log.

The app no longer asks the user to type in five numbers. The model needs 25
features built from ~50 days of history, so hand entry is not possible and was
never a good demo anyway. Instead this module pulls the latest completed session
from Yahoo, builds features from real history, and predicts the next session.

Every live prediction is written to SQLite before it is shown. Later, once the
market has actually closed on the day being predicted, resolve_pending() fills
in what really happened. That is what /track reports: not the backtest, but how
this deployment has done on days it had never seen at training time.
"""

import os
import json
import sqlite3
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

from features import build_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
DB_PATH = os.path.join(BASE_DIR, 'predictions.db')
PRED_CSV = os.path.join(BASE_DIR, 'test_predictions.csv')
BACKTEST_JSON = os.path.join(BASE_DIR, 'backtest.json')
CACHE_DIR = os.path.join(BASE_DIR, 'data')

# The bundle carries the feature order with the model, so the app can never feed
# values into the wrong slots.
bundle = joblib.load(MODEL_PATH)
model = bundle['model']
return_model = bundle.get('return_model')
FEATURES = bundle['features']
TICKER = bundle['ticker']

# Yahoo is rate limited and occasionally down. Serve a short-lived in-process
# cache so a page refresh does not trigger a new download.
_CACHE_TTL_SECONDS = 900
_cache = {'data': None, 'fetched_at': None, 'source': None}
_lock = threading.Lock()


# ------------------------------------------------------------------ data access

def _cache_file():
    return os.path.join(CACHE_DIR, TICKER.replace('.', '_') + '.csv')


def _read_csv_fallback():
    path = _cache_file()
    if not os.path.exists(path):
        raise RuntimeError('No cached price data at ' + path)
    return pd.read_csv(path, index_col=0, parse_dates=True).dropna()


def fetch_live(force=False):
    """
    Most recent ~2 years of daily OHLCV. Falls back to the CSV cache when Yahoo
    is unreachable, so the app still renders offline rather than 500-ing.
    """
    with _lock:
        now = datetime.now(timezone.utc)
        fresh = (_cache['data'] is not None and _cache['fetched_at'] is not None
                 and (now - _cache['fetched_at']).total_seconds() < _CACHE_TTL_SECONDS)
        if fresh and not force:
            return _cache['data'], _cache['source']

        source = 'live'
        try:
            import yfinance as yf
            data = yf.download(TICKER, period='2y', auto_adjust=True,
                               progress=False, threads=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.dropna()
            if data.empty:
                raise RuntimeError('empty frame from Yahoo')
            # Keep the on-disk cache warm for the trainer and for offline runs.
            try:
                existing = _read_csv_fallback()
                merged = pd.concat([existing, data])
                merged = merged[~merged.index.duplicated(keep='last')].sort_index()
                merged.to_csv(_cache_file())
            except Exception:
                pass
        except Exception as exc:
            data = _read_csv_fallback()
            source = 'cache (' + type(exc).__name__ + ')'

        _cache.update({'data': data, 'fetched_at': now, 'source': source})
        return data, source


# -------------------------------------------------------------------- inference

def predict_latest(force=False):
    """
    Predict the next trading session from the most recent completed session.

    Returns a dict describing the signal, or raises RuntimeError if there is not
    enough history to build features.
    """
    data, source = fetch_live(force=force)
    feats = build_features(data).dropna()
    if feats.empty:
        raise RuntimeError('Not enough history to build features.')

    latest_date = feats.index[-1]
    row = feats.iloc[[-1]][FEATURES]

    proba_up = float(model.predict_proba(row)[0, 1])
    direction = int(proba_up > 0.5)
    pred_return = float(return_model.predict(row)[0]) if return_model is not None else float('nan')

    last = data.loc[latest_date]
    close_now = float(last['Close'])

    return {
        'ticker': TICKER,
        'source': source,
        'signal_date': latest_date.date().isoformat(),
        'close': round(close_now, 2),
        'open': round(float(last['Open']), 2),
        'high': round(float(last['High']), 2),
        'low': round(float(last['Low']), 2),
        'volume': int(last['Volume']),
        'proba_up': round(proba_up * 100, 1),
        'direction': direction,
        'direction_label': 'UP' if direction else 'DOWN / FLAT',
        'confidence': round(abs(proba_up - 0.5) * 200, 1),
        'pred_return_pct': round(pred_return * 100, 3),
        'implied_close': round(close_now * (1 + pred_return), 2),
    }


def model_info():
    """Details shown on the page so it is clear what the prediction is based on."""
    m = bundle['metrics']
    return {
        'ticker': TICKER,
        'model_name': bundle['model_name'],
        'trained_at': bundle['trained_at'],
        'target': 'next-session direction',
        'n_features': len(FEATURES),
        'accuracy': round(m['test']['accuracy'], 2),
        'auc': round(m['test']['auc'], 4),
        'baseline': round(m['baseline_always_up_test'], 2),
        'edge': round(m['test']['accuracy'] - m['baseline_always_up_test'], 2),
        'train_end': bundle['train_end'],
        'test_start': bundle['test_start'],
        'test_end': bundle['test_end'],
    }


# ------------------------------------------------------------- prediction log

def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker           TEXT    NOT NULL,
                signal_date      TEXT    NOT NULL,
                made_at          TEXT    NOT NULL,
                close_at_signal  REAL    NOT NULL,
                proba_up         REAL    NOT NULL,
                pred_direction   INTEGER NOT NULL,
                pred_return_pct  REAL,
                model_name       TEXT,
                actual_close     REAL,
                actual_return_pct REAL,
                actual_direction INTEGER,
                correct          INTEGER,
                resolved_at      TEXT,
                UNIQUE(ticker, signal_date)
            )
        """)


def log_prediction(p):
    """
    Record a live prediction. One row per (ticker, signal_date): refreshing the
    page must not stuff the log with duplicates and inflate the hit rate.
    """
    init_db()
    with _connect() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO predictions
                (ticker, signal_date, made_at, close_at_signal, proba_up,
                 pred_direction, pred_return_pct, model_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (p['ticker'], p['signal_date'],
              datetime.now(timezone.utc).isoformat(timespec='seconds'),
              p['close'], p['proba_up'] / 100.0, p['direction'],
              p['pred_return_pct'], bundle['model_name']))


def resolve_pending():
    """
    Score any logged prediction whose outcome is now known.

    A prediction made on signal_date is about the next session. Once price data
    exists for a date strictly after signal_date, the answer is available.
    """
    init_db()
    with _connect() as conn:
        pending = conn.execute(
            'SELECT * FROM predictions WHERE correct IS NULL ORDER BY signal_date'
        ).fetchall()
        if not pending:
            return 0

        try:
            data, _ = fetch_live()
        except Exception:
            return 0

        closes = data['Close']
        resolved = 0
        for row in pending:
            sig = pd.Timestamp(row['signal_date'])
            later = closes[closes.index > sig]
            if later.empty:
                continue                      # the market has not closed yet
            actual_close = float(later.iloc[0])
            actual_ret = actual_close / row['close_at_signal'] - 1.0
            actual_dir = int(actual_ret > 0)
            conn.execute("""
                UPDATE predictions
                   SET actual_close = ?, actual_return_pct = ?, actual_direction = ?,
                       correct = ?, resolved_at = ?
                 WHERE id = ?
            """, (actual_close, actual_ret * 100, actual_dir,
                  int(actual_dir == row['pred_direction']),
                  datetime.now(timezone.utc).isoformat(timespec='seconds'),
                  row['id']))
            resolved += 1
        return resolved


def track_record(limit=100):
    """Everything /track needs: the log, plus how it has actually done."""
    init_db()
    resolve_pending()
    with _connect() as conn:
        rows = [dict(r) for r in conn.execute(
            'SELECT * FROM predictions ORDER BY signal_date DESC LIMIT ?', (limit,)
        ).fetchall()]
        agg = dict(conn.execute("""
            SELECT COUNT(*)                                  AS total,
                   SUM(CASE WHEN correct IS NULL THEN 1 ELSE 0 END) AS pending,
                   SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END)     AS hits,
                   SUM(CASE WHEN correct IS NOT NULL THEN 1 ELSE 0 END) AS scored
              FROM predictions
        """).fetchone())

    scored = agg['scored'] or 0
    hits = agg['hits'] or 0
    return {
        'rows': rows,
        'total': agg['total'] or 0,
        'pending': agg['pending'] or 0,
        'scored': scored,
        'hits': hits,
        'hit_rate': round(hits / scored * 100, 1) if scored else None,
    }


# --------------------------------------------------------------- chart payloads

def chart_history(days=90):
    """
    Actual vs predicted over the tail of the held-out test set.

    This is out-of-sample: these days were never seen during training or model
    selection, so the chart is a fair picture of behaviour, not a fit.
    """
    if not os.path.exists(PRED_CSV):
        return None
    df = pd.read_csv(PRED_CSV, parse_dates=['date']).tail(days)

    # The predicted close for day t+1, drawn against the close that happened.
    pred_close = df['close'] * (1 + df['pred_return'])
    actual_next = df['close'] * (1 + df['actual_return'])

    return {
        'dates': [d.date().isoformat() for d in df['date']],
        'actual_close': [round(float(x), 2) for x in actual_next],
        'predicted_close': [round(float(x), 2) for x in pred_close],
        'proba_up': [round(float(x) * 100, 1) for x in df['pred_proba_up']],
        'correct': [int(a == p) for a, p in
                    zip(df['actual_direction'], df['pred_direction'])],
        'hit_rate': round(float((df['actual_direction'] == df['pred_direction']).mean() * 100), 1),
    }


def backtest_summary():
    """The saved backtest, trimmed to what the dashboard renders."""
    if not os.path.exists(BACKTEST_JSON):
        return None
    with open(BACKTEST_JSON) as f:
        bt = json.load(f)
    return {
        'window': bt['window'],
        'assumptions': bt['assumptions'],
        'dates': bt['dates'],
        'strategy_equity': bt['strategy_net']['equity_curve'],
        'gross_equity': bt['strategy_gross']['equity_curve'],
        'buyhold_equity': bt['buy_hold']['equity_curve'],
        'strategy': {k: v for k, v in bt['strategy_net'].items() if k != 'equity_curve'},
        'gross': {k: v for k, v in bt['strategy_gross'].items() if k != 'equity_curve'},
        'buyhold': {k: v for k, v in bt['buy_hold'].items() if k != 'equity_curve'},
        'cost_drag_pct': round(bt['cost_drag_pct'], 2),
        'sensitivity': bt['sensitivity'],
    }
