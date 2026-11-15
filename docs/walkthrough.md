# Walkthrough

This follows one run of the pipeline on the default synthetic log (seed 7), function by function. Every number here
comes from running the code in this repo. You can reproduce it in a Python shell started at the repo root with
`import sys; sys.path.insert(0, "src")`.

## 1. The raw fill log

```python
from metalabel.data import make_synthetic_log, pair_trades
log = make_synthetic_log()
```

`log` has 3,000 rows: one Entry and one Exit row for each of 1,500 positions. The columns match a broker export
(`Position_ID`, `Trade_Date`, `Trade_Time`, `Price`, `Take_Profit`, `Stop_Loss`, `Entry_Exit`, `Realized_PnL`,
`Notes`, ...). `python scripts/make_synthetic_data.py` writes the same log to `data/sample_trades.csv`.

## 2. One row per trade

```python
trades = pair_trades(log)
```

1,500 closed trades, entered between January 2016 and August 2024. Each row has `entry_dt`, `entry_price`,
`take_profit`, `exit_dt`, `exit_price`, `pnl` and `exit_reason`. Open positions would be dropped here.

## 3. Entry-time features

```python
from metalabel.features import build_feature_table
table = build_feature_table(trades, mode="t1")
```

For every trade, `prior_history` picks the closed trades the model could have seen. In `t1` mode that means
trades with `exit_dt` before the entry's calendar day. `entry_features` then builds the 16 features listed in
`FEATURES`. Trades without enough history are dropped, which leaves 1,488 rows. The base win rate is 45.8%.

## 4. Walk-forward

```python
from metalabel.walkforward import walk_forward
oos = walk_forward(table, "random_forest", keep_frac=0.75)
```

2016 and 2017 are warm-up. For each test year from 2018, the model is fitted on earlier trades that had already
closed, and the cutoff is the 25th percentile of its probabilities on that training set.

| Test year | Trades | Cutoff | Share kept |
|---|---|---|---|
| 2018 | 172 | 0.368 | 79.7% |
| 2019 | 187 | 0.331 | 99.5% |
| 2020 | 167 | 0.388 | 79.0% |
| 2021 | 174 | 0.407 | 33.3% |
| 2022 | 182 | 0.325 | 80.2% |
| 2023 | 179 | 0.316 | 86.0% |
| 2024 | 105 | 0.326 | 81.0% |

The share kept moves a lot from year to year. That is expected: the cutoff is fixed before the year starts, and the
test-year scores do not follow the training distribution exactly. Picking the cutoff from the test year would
always give 75%, but it would use information from the future.

## 5. Baseline against filtered

```python
from metalabel.metrics import compare
compare(oos)
```

Both rows are measured on the same business-day calendar. For the random forest the filter keeps 77% of the 1,166
out-of-sample trades, the win rate goes from 45.5% to 53.3%, Sharpe from -0.49 to 0.49, and max drawdown from
-18.3% to -9.8%. `scripts/run_backtest.py` prints this table for all three models and writes it to `results/`.

## 6. Serving

`scripts/train_bundle.py` fits one model on the whole table and fixes the cutoff from the same training
probabilities. On the default log it trains on 1,488 trades and sets the cutoff at 0.285. `api/app.py` loads that
bundle: `/ingest` stores a caller's closed trades, and `/score` applies the same T-1 rule to score a new trade.

## Things to try

- Build the table with `mode="sequential"`, run `walk_forward` again and compare the results.
- Build a log with `make_synthetic_log(signal_strength=0)` and confirm the filter finds no edge.
- Change `--keep` once, before looking at results, and see how the keep rate and drawdown move.
