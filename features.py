"""
features.py - feature engineering and target construction.

Why this file exists at all: the earlier version of this project fed raw Open,
High, Low, Close and Volume straight into a regressor and predicted the next
day's *price level*. That looks impressive (r2 ~ 0.99) and means nothing, because
tomorrow's price is ~99% explained by today's price. The naive "tomorrow closes
where today closed" baseline scores the same, so the model adds nothing.

The fix is to change the question. Here the target is the next day's *return*,
which is roughly stationary and has no free lunch hiding in it. Raw price levels
are useless for that - a level of 1500 tells you nothing about tomorrow's percent
move - so the model needs stationary features instead: returns, volatility,
momentum and normalised volume.

Every feature is computed from data available at or before the close of day t.
The target is the move from close(t) to close(t+1). Nothing looks ahead.
"""

import numpy as np
import pandas as pd

OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']


# ------------------------------------------------------------ indicator helpers

def rsi(close, window=14):
    """Relative Strength Index. Wilder's smoothing, expressed 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd_histogram(close, fast=12, slow=26, signal=9):
    """MACD line minus its signal line - momentum, already differenced."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    # Scale by price so the value is comparable across the 350 -> 1900 range.
    return (macd_line - signal_line) / close


def bollinger_pct_b(close, window=20, n_std=2):
    """Where the close sits inside its Bollinger band. 0 = lower, 1 = upper."""
    mid = close.rolling(window).mean()
    sd = close.rolling(window).std()
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    return (close - lower) / (upper - lower).replace(0, np.nan)


# ------------------------------------------------------------ feature assembly

def build_features(data):
    """Turn an OHLCV frame into stationary features indexed by the same dates."""
    close = data['Close']
    high = data['High']
    low = data['Low']
    open_ = data['Open']
    volume = data['Volume']

    ret = close.pct_change()

    f = pd.DataFrame(index=data.index)

    # Momentum: where has it come from over several horizons.
    for lag in (1, 2, 3, 5, 10):
        f[f'ret_lag_{lag}'] = ret.shift(lag - 1)

    # Trend and volatility over rolling windows.
    for w in (5, 10, 20):
        f[f'ret_mean_{w}'] = ret.rolling(w).mean()
        f[f'ret_std_{w}'] = ret.rolling(w).std()

    # Distance from moving averages, as a fraction of price.
    for w in (5, 20, 50):
        f[f'dist_sma_{w}'] = close / close.rolling(w).mean() - 1

    # Classic technical indicators.
    f['rsi_14'] = rsi(close) / 100.0
    f['macd_hist'] = macd_histogram(close)
    f['bb_pct_b'] = bollinger_pct_b(close)

    # Intraday shape of the current bar.
    f['hl_range'] = (high - low) / close
    f['close_position'] = (close - low) / (high - low).replace(0, np.nan)
    f['gap_open'] = open_ / close.shift(1) - 1
    f['body'] = (close - open_) / close

    # Volume, normalised so 2015 volumes are comparable with 2026 volumes.
    log_vol = np.log1p(volume)
    f['vol_z_20'] = (log_vol - log_vol.rolling(20).mean()) / log_vol.rolling(20).std()
    f['vol_ratio_5'] = volume / volume.rolling(5).mean()

    # Calendar effects.
    f['day_of_week'] = data.index.dayofweek
    f['month'] = data.index.month

    return f.replace([np.inf, -np.inf], np.nan)


FEATURE_NAMES = None  # populated on first build_features call via build_dataset


def build_dataset(data):
    """
    Returns X, y_return, y_direction, close.

    y_return    - fractional move from close(t) to close(t+1), e.g. 0.012 = +1.2%
    y_direction - 1 if that move is positive, else 0
    close       - close(t), kept aligned for backtesting
    """
    f = build_features(data)

    next_ret = data['Close'].pct_change().shift(-1)

    frame = f.copy()
    frame['_target_ret'] = next_ret
    frame['_close'] = data['Close']
    frame = frame.dropna()

    feature_cols = [c for c in frame.columns if not c.startswith('_')]

    X = frame[feature_cols]
    y_ret = frame['_target_ret']
    y_dir = (y_ret > 0).astype(int)
    close = frame['_close']

    return X, y_ret, y_dir, close
