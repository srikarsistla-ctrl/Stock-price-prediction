"""
backtest.py - turn model predictions into a trading strategy and score it honestly.

A classification accuracy of 53.7% sounds like an edge. This file asks the only
question that matters: does that edge survive contact with transaction costs?

Strategy
    Long/flat. At the close of day t the model emits P(up) for day t+1.
    If P(up) > threshold, hold the stock through day t+1. Otherwise hold cash.
    There is no shorting and no leverage.

Alignment
    test_predictions.csv stores, on row t: the close on day t, the prediction
    made at the close of day t, and actual_return = the close(t) -> close(t+1)
    move. So the position taken on row t earns actual_return on row t. Nothing
    looks ahead.

Costs
    COST_BPS is charged per side on every change in position. A round trip
    (buy then sell) therefore costs 2 x COST_BPS. 10 bps a side is a realistic
    all-in retail figure for Indian equities once brokerage, STT, exchange
    charges, stamp duty and slippage are counted.

Run:  python backtest.py
"""

import os
import json

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_PATH = os.path.join(BASE_DIR, 'test_predictions.csv')
OUT_PATH = os.path.join(BASE_DIR, 'backtest.json')

COST_BPS = 10.0          # per side, in basis points. 10 bps = 0.10%
TRADING_DAYS = 252
THRESHOLD = 0.50         # P(up) above this means hold the stock


# ------------------------------------------------------------------- statistics

def max_drawdown(equity):
    """Largest peak-to-trough fall in the equity curve, as a negative fraction."""
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def summarise(name, daily_returns, positions=None, n_trades=None):
    """Standard performance stats for a daily return series."""
    r = np.asarray(daily_returns, dtype=float)
    equity = np.cumprod(1.0 + r)
    n = len(r)
    years = n / TRADING_DAYS

    total = float(equity[-1] - 1.0)
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) if years > 0 else float('nan')
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if n > 1 else float('nan')
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else float('nan')

    downside = r[r < 0]
    sortino = (float(r.mean() / downside.std(ddof=1) * np.sqrt(TRADING_DAYS))
               if len(downside) > 1 and downside.std(ddof=1) > 0 else float('nan'))

    mdd = max_drawdown(equity)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else float('nan')

    active = r[r != 0]
    win_rate = float((active > 0).mean() * 100) if len(active) else float('nan')

    return {
        'name': name,
        'total_return_pct': total * 100,
        'cagr_pct': cagr * 100,
        'volatility_pct': vol * 100,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown_pct': mdd * 100,
        'calmar': calmar,
        'win_rate_pct': win_rate,
        'exposure_pct': float(np.mean(positions) * 100) if positions is not None else 100.0,
        'n_trades': int(n_trades) if n_trades is not None else 0,
        'n_days': n,
        'final_equity': float(equity[-1]),
        'equity_curve': [round(float(x), 6) for x in equity],
    }


# --------------------------------------------------------------------- strategy

def run_strategy(df, threshold=THRESHOLD, cost_bps=COST_BPS):
    """Long/flat on the model signal, net of per-side transaction costs."""
    signal = (df['pred_proba_up'] > threshold).astype(float).values
    market = df['actual_return'].values

    # Position changes are charged. The first day charges for entering.
    prev = np.concatenate([[0.0], signal[:-1]])
    turnover = np.abs(signal - prev)
    cost = turnover * (cost_bps / 10000.0)

    gross = signal * market
    net = gross - cost

    return net, gross, signal, int(turnover.sum())


def main():
    if not os.path.exists(PRED_PATH):
        raise SystemExit('Missing ' + PRED_PATH + '. Run train.py first.')

    df = pd.read_csv(PRED_PATH, parse_dates=['date'])
    print('')
    print('Backtest window : %s -> %s  (%d trading days)'
          % (df['date'].iloc[0].date(), df['date'].iloc[-1].date(), len(df)))
    print('Cost assumption : %.0f bps per side (%.0f bps round trip)'
          % (COST_BPS, 2 * COST_BPS))
    print('Signal          : long when P(up) > %.2f, otherwise flat' % THRESHOLD)
    print('')

    net, gross, signal, n_trades = run_strategy(df)

    strat_net = summarise('Model strategy (net of costs)', net, signal, n_trades)
    strat_gross = summarise('Model strategy (gross, no costs)', gross, signal, n_trades)
    buy_hold = summarise('Buy and hold', df['actual_return'].values)

    header = ('%-34s %10s %9s %8s %9s %8s %8s'
              % ('', 'total %', 'CAGR %', 'Sharpe', 'maxDD %', 'expo %', 'trades'))
    print(header)
    print('-' * len(header))
    for s in (buy_hold, strat_gross, strat_net):
        print('%-34s %10.2f %9.2f %8.2f %9.2f %8.1f %8d'
              % (s['name'], s['total_return_pct'], s['cagr_pct'], s['sharpe'],
                 s['max_drawdown_pct'], s['exposure_pct'], s['n_trades']))

    cost_drag = strat_gross['total_return_pct'] - strat_net['total_return_pct']
    print('')
    print('Transaction costs consumed %.2f percentage points of return across %d trades.'
          % (cost_drag, n_trades))

    if strat_net['sharpe'] > buy_hold['sharpe']:
        print('Net of costs the strategy has a HIGHER Sharpe than buy and hold')
        print('(%.2f vs %.2f) - it is in the market only %.0f%% of the time, so it'
              % (strat_net['sharpe'], buy_hold['sharpe'], strat_net['exposure_pct']))
        print('takes less risk to get its return.')
    else:
        print('Net of costs the strategy does NOT beat buy and hold on Sharpe')
        print('(%.2f vs %.2f). This is the usual outcome and the honest headline:'
              % (strat_net['sharpe'], buy_hold['sharpe']))
        print('a small classification edge is not automatically a tradeable one.')

    # ---- how sensitive is the result to the cost assumption and threshold? ---
    print('')
    print('Sensitivity - total return %, by signal threshold and cost per side:')
    thresholds = [0.45, 0.50, 0.52, 0.55, 0.60]
    costs = [0.0, 5.0, 10.0, 20.0]
    print('%-12s' % 'threshold' + ''.join('%12s' % ('%.0f bps' % c) for c in costs))
    grid = []
    for t in thresholds:
        row = {'threshold': t, 'by_cost': {}}
        line = '%-12.2f' % t
        for c in costs:
            n, _, sig, _ = run_strategy(df, threshold=t, cost_bps=c)
            tot = float(np.cumprod(1.0 + n)[-1] - 1.0) * 100
            row['by_cost'][str(c)] = tot
            line += '%12.2f' % tot
        grid.append(row)
        print(line)
    print('%-12s' % 'buy & hold' + '%12.2f' % buy_hold['total_return_pct'])

    payload = {
        'window': {'start': str(df['date'].iloc[0].date()),
                   'end': str(df['date'].iloc[-1].date()),
                   'days': len(df)},
        'assumptions': {'cost_bps_per_side': COST_BPS, 'threshold': THRESHOLD,
                        'strategy': 'long/flat, no leverage, no shorting'},
        'dates': [str(d.date()) for d in df['date']],
        'strategy_net': strat_net,
        'strategy_gross': strat_gross,
        'buy_hold': buy_hold,
        'cost_drag_pct': cost_drag,
        'sensitivity': grid,
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(payload, f, indent=2)
    print('')
    print('Saved ' + OUT_PATH)


if __name__ == '__main__':
    main()
