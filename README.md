# Next-Session Direction Model — INFY.NS

A daily equity direction predictor built to **measure** how much signal exists rather than
manufacture an impressive-looking number. The interesting result is negative, and that is
the point.

## The story

| Version | Target | Headline | Why it was wrong |
|---|---|---|---|
| v1 (notebook) | today's close from today's high/low/open | R² = 0.9999 | **Target leakage** — the answer was inside the inputs. High and low are only known after the close. It also used a shuffled `train_test_split`, scattering future days into training on time-series data. |
| v2 | tomorrow's close from today's OHLCV | RMSE 24.48 | Honest, but **pointless**. The naive "tomorrow closes where today closed" baseline scored RMSE 24.50. A price level is ~99% explained by the previous price level, so R² near 1.0 measures nothing. |
| v3 (current) | tomorrow's **direction** and **return** | 53.0% vs 49.3% baseline | Finally a benchmark a model has to earn. Then the backtest asks whether that edge survives costs. It does not. |

## Results

**Direction classification** (held-out test, 2024-05-02 → 2026-08-12, 564 sessions)

| | Accuracy | AUC |
|---|---|---|
| Always predict "up" (baseline) | 49.29% | — |
| Logistic regression, 25 features | **53.01%** | 0.546 |

Model chosen on a separate **validation** block and scored **once** on test. Selecting the
winner by test score would have leaked the test set into that decision.

**Backtest** — long/flat, no shorting, 10 bps per side

| | Total return | CAGR | Sharpe | Max DD | Exposure | Trades |
|---|---|---|---|---|---|---|
| Buy & hold | −16.0% | −7.4% | −0.16 | −50.1% | 100% | 0 |
| Strategy, gross | −12.8% | −5.9% | −0.13 | −37.3% | 65% | 189 |
| Strategy, **net of costs** | **−27.7%** | −13.5% | −0.48 | −41.9% | 65% | 189 |

**The finding:** a 3.7-point classification edge is real but flips often enough to trade 189
times in 564 sessions. At a realistic 10 bps per side that costs **14.9 percentage points**
and erases the advantage. The model does reduce drawdown (−37% vs −50%) and sits in cash
35% of the time, but it does not earn its transaction costs.

`backtest.py` prints a sensitivity grid across signal thresholds and cost assumptions so the
decay is visible rather than asserted.

## Method

- **Stationary features only** (`features.py`) — 25 of them: lagged returns, rolling
  mean/std of returns, distance from 5/20/50-day SMAs, RSI(14), MACD histogram, Bollinger
  %B, intraday range and body, gap-open, volume z-score, calendar effects. Raw price levels
  are excluded: a level of 1500 says nothing about tomorrow's percent move.
- **No lookahead.** Every feature uses data available at or before the close of day *t*;
  the target is the close(*t*) → close(*t*+1) move.
- **Chronological three-way split** — 60% train / 20% validation / 20% test. Train on the
  past, select on the middle, report on the future.
- **Costs charged on every position change**, not just round trips.
- **Live predictions are logged before they are displayed**, keyed by session date so a page
  refresh cannot create a second attempt, and resolved automatically against real closes.

## Running it

```bash
pip install -r requirements.txt
python train.py       # fits, selects on validation, scores on test, writes model.pkl
python backtest.py    # trading simulation + cost sensitivity -> backtest.json
python app.py         # dashboard at http://127.0.0.1:5000
```

Deploy: `gunicorn app:app` (see `Procfile`).

## App

- `/` — live signal auto-fetched from Yahoo (no manual data entry), out-of-sample
  predicted-vs-actual chart, backtest equity curves, cost sensitivity table.
- `/track` — every prediction this deployment has made, scored against what actually
  happened. A backtest can be tuned until it looks good; a forward log cannot.
- `/api/prediction`, `/api/track` — JSON.

Falls back to the cached CSV and labels the data source when Yahoo is unreachable, so the
page renders offline instead of erroring.

## Files

| File | Purpose |
|---|---|
| `features.py` | Feature engineering and target construction |
| `train.py` | Training, model selection, evaluation |
| `backtest.py` | Trading simulation, costs, sensitivity analysis |
| `utils.py` | Live data, inference, SQLite prediction log |
| `app.py` | Flask routes |
| `train_price_level_legacy.py` | The v2 price-level trainer, kept for comparison |
| `metrics.json`, `backtest.json`, `test_predictions.csv` | Generated artifacts |

## Caveats

Single ticker, single regime, no walk-forward retraining, no shorting, and the test window
happens to be a drawdown period for INFY. The 53% accuracy is one number from one split —
treat it as an upper bound. This is a modelling and evaluation exercise, **not investment
advice**.
