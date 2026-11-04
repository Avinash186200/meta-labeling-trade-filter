# Methodology and leakage audit

## Problem

A base strategy produces trades. Some of them are predictably weak. Meta-labeling (López de Prado, *Advances in Financial Machine Learning*, ch. 3) trains a secondary classifier to predict whether a given primary signal will be profitable, and uses that prediction to size or skip the trade.

Here the label is `win = pnl > 0`, and the action is binary: take or skip.

## Data contract

| Field | Used for |
|---|---|
| `Position_ID` | Pairs the Entry and Exit rows |
| `Trade_Date`, `Trade_Time` | Entry and exit timestamps |
| `Price`, `Take_Profit` | Entry price and take-profit distance |
| `Realized_PnL` (Exit row) | Label, and the history features of later trades |
| `Notes` (Exit row) | Exit reason. Kept for analysis, not used as a feature |

`Stop_Loss` was equal to the entry price on every row of the original log, so it carried no information and is not used.

## Feature timing

There are two history modes:

- `sequential`: every trade with `exit_dt <= entry_dt`. This is only valid if fills reach the model in real time.
- `t1` (the default for all reported results): only trades with `exit_dt` before the entry's calendar day. This is the right mode when the trade log is uploaded once a day.

## Validation design

- Yearly expanding walk-forward. The first two years are training-only.
- **Purge at the fold boundary**: a trade is only used for training if it had closed before the test year starts, so no label from inside the test year is used.
- The scaler and model are fitted inside each fold, through an sklearn `Pipeline`.
- **Cutoff**: the `(1 - keep)` quantile of the model's probabilities on that fold's **training** set. The original research code took this quantile from the test year, which gives the filter a peek at the future score distribution.
- **Keep-fraction** is fixed before running. The original code compared 0.90, 0.75, 0.55 and 0.50 on the same out-of-sample data and favoured the best. That is a multiple-testing problem, so it was removed.
- Baseline and filtered equity curves are measured over the same business-day calendar.

## Checks in CI

- Pairing gives exactly one row per position, and open positions are dropped.
- In T-1 mode, the history never contains a same-day exit.
- No training trade is still open when its test year begins.
- Each test year has exactly one cutoff, set before that year is scored.
- **Noise test**: on a log with no signal, the filter's win-rate lift stays under 8 percentage points.
- The scoring function works end to end on a new trade.

## Known limitations

- Only one instrument and one strategy were studied. The features are generic, but nothing here shows that the result transfers to other instruments or strategies.
- There is no transaction-cost adjustment for skipped trades. Skipping a signal is treated as free, which holds for a filter that simply does not send the order.
- The real-data results have not been run through a final untouched hold-out, so they are not published.
