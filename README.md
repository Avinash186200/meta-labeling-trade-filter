# Meta-Labeling Trade Filter

[![CI](https://github.com/Avinash186200/meta-labeling-trade-filter/actions/workflows/ci.yml/badge.svg)](https://github.com/Avinash186200/meta-labeling-trade-filter/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.11-blue) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A second-stage model that sits on top of an existing trading strategy, scores every trade at entry, and skips the weakest ones. The base strategy decides *what* to trade. This model only decides *whether to take it*.

I built this for a long-only trend strategy on a futures contract. The original trade log is proprietary, so it is not in this repo. Everything here runs on a **synthetic trade log that has exactly the same columns**, so you can clone the repo and reproduce every number below.

## At a glance

| | |
|---|---|
| **Input** | Broker-style fill log: one Entry row and one Exit row per position (`Position_ID`, `Trade_Date`, `Trade_Time`, `Price`, `Take_Profit`, `Realized_PnL`, ...) |
| **Method** | Entry-time features from the trade's own structure and prior closed trades, then a yearly expanding walk-forward with LogReg, RandomForest and gradient boosting |
| **Output** | Out-of-sample P(win) per trade, a take/skip decision, baseline-vs-filtered statistics, and a deployable scoring bundle plus API |
| **Leakage controls** | T-1 strict history, a cutoff fixed from training data only, keep-fraction chosen in advance, and a noise test in CI |
| **Status** | Research complete. Results below are synthetic and illustrative only |

## How it works

```
fill log ──► pair entries/exits ──► entry-time features ──► walk-forward model ──► P(win) ──► take / skip
 (CSV)        data.py               features.py             walkforward.py         scoring.py   api/
```

1. **Pair** the Entry and Exit rows into one row per closed trade. Positions that are still open are dropped.
2. **Features** use only what was known before entry:
   - the take-profit distance
   - time of day, day of week and month
   - the gap from the last exit
   - the last trade's P&L and duration
   - rolling win rate and average P&L over 5, 10 and 20 trades, P&L volatility, and the current streak
3. **Walk-forward**, year by year: train on all earlier years and test on the next one. The first two years are warm-up only.
4. **Filter**: keep the trades whose P(win) is at or above a cutoff. The cutoff is a quantile of the model's probabilities on the **training** data.

## Results on synthetic data

`python scripts/run_backtest.py` (seed 7, 1,500 trades, about 1,166 out-of-sample, keeping the top 75%):

| Model | Trades kept | Win rate | Sharpe | Max DD |
|---|---|---|---|---|
| Baseline (take all) | 100% | 45.5% | -0.49 | -18.3% |
| LogReg filter | 69% | 56.5% | 0.33 | -10.8% |
| RandomForest filter | 77% | 53.3% | 0.49 | -9.8% |
| Gradient boosting filter | 76% | 53.0% | 0.33 | -11.1% |

The realised keep rate is not exactly 75%: the cutoff is a quantile of *training* probabilities, and the test-year score distribution drifts from it. That gap is the honest price of not peeking at the test year.

The synthetic generator has a known, modest signal planted in the take-profit distance. The point of these numbers is that the pipeline **recovers a real signal without inventing one**. With the signal switched off (`signal_strength=0`), the filter shows no edge. That check is a test in CI.

> **About the real-data result.** On the proprietary trade log, the filter improved win rate and drawdown out of sample. I don't publish those figures here: they come from a single strategy and a single instrument, and they have not been through a final hold-out. Treat this repo as a method, not a performance claim.

## Leakage audit

These were found and fixed during the research. The details are in [docs/methodology.md](docs/methodology.md).

| Issue | Effect | Fix |
|---|---|---|
| Cutoff picked from the **test** year's probabilities | Filter "knew" the future score distribution | Cutoff taken from the training probabilities |
| Rolling features included trades closed **the same day**, which a daily-upload workflow would not have yet | Optimistic features | `t1` mode: prior calendar days only (default) |
| Several keep-fractions tried, best one reported | Selection bias | One keep-fraction, fixed in advance (`--keep 0.75`) |
| Previous-trade duration computed from the exit **date** | Wrong feature | Uses exit **timestamp** |
| Training fold included trades entered before the test year but **closed inside it** | Label not yet known at the fold boundary | Training rows must have `exit_dt` before the test year starts (purge); tested in CI |

## Repo layout

```
src/metalabel/   data.py · features.py · walkforward.py · metrics.py · scoring.py
scripts/         run_backtest.py · make_synthetic_data.py
api/             app.py  (FastAPI: /health, /ingest, /score)
tests/           pairing, T-1 boundary, fold purge, cutoff, noise test, scoring, API auth
docs/            methodology.md
```

## Quick start

```bash
pip install -r requirements.txt
pytest -q
python scripts/run_backtest.py                 # synthetic
python scripts/run_backtest.py --trades your_fills.csv --keep 0.75
```

To serve a model, train a bundle with `metalabel.scoring.train_bundle`, save it, then run:

```bash
MODEL_BUNDLE=model.joblib API_KEYS=change-me uvicorn api.app:app
```

## Original research code

[`original/`](original/README.md) holds the code as I first wrote it for this work, sanitised and nothing else:
client, company and person names replaced, credentials and local paths removed, data files and notebook outputs
dropped, and internal deployment notes, client-facing decks and client findings memos left out. It is kept so
the real work can be read alongside the maintained version. Some of it contains the issues
described above; comparing the two shows each fix.

## License

MIT

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [CHANGELOG](CHANGELOG.md).
