"""Entry-time features, built only from information available before entry.

Two modes:
  * "sequential" - uses every trade closed before this one's entry, including
    same-day trades. Correct only if fills are known intraday in real time.
  * "t1" (default) - strict: uses only trades closed on a PRIOR calendar day.
    This matches a daily-upload workflow and is the mode used for results.

Fix vs the original research code: previous-trade duration is computed from
exit *timestamp* (not exit date) minus entry timestamp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "target_dist_pct", "entry_hour", "entry_dow", "entry_month",
    "gap_pct", "days_since_last_exit", "prev_trade_pnl", "prev_duration_min",
    "roll_winrate_5", "roll_avgpnl_5", "roll_winrate_10", "roll_avgpnl_10",
    "roll_winrate_20", "roll_avgpnl_20", "roll_vol_10", "streak",
]


def _streak(wins: np.ndarray) -> int:
    """Length of the current win (+) or loss (-) run at the end of `wins`."""
    if len(wins) == 0:
        return 0
    last, s = wins[-1], 0
    for w in wins[::-1]:
        if w != last:
            break
        s += 1
    return s if last == 1 else -s


def history_features(prior: pd.DataFrame) -> dict:
    """Features derived from a history of CLOSED trades (sorted by exit time)."""
    out = {k: np.nan for k in FEATURES[4:]}
    if prior.empty:
        return out
    last = prior.iloc[-1]
    pnl = prior["pnl"].to_numpy()
    win = (pnl > 0).astype(int)
    out["prev_trade_pnl"] = float(last["pnl"])
    out["prev_duration_min"] = (last["exit_dt"] - last["entry_dt"]).total_seconds() / 60
    for n in (5, 10, 20):
        if len(pnl) >= max(3, n // 2):
            out[f"roll_winrate_{n}"] = win[-n:].mean()
            out[f"roll_avgpnl_{n}"] = pnl[-n:].mean()
    if len(pnl) >= 5:
        out["roll_vol_10"] = pnl[-10:].std(ddof=1)
    out["streak"] = _streak(win)
    out["_last_exit_price"] = float(last["exit_price"])
    out["_last_exit_dt"] = last["exit_dt"]
    return out


def entry_features(entry_dt, entry_price: float, take_profit: float, prior: pd.DataFrame) -> dict:
    """Full feature vector for one trade, given the history visible at entry."""
    entry_dt = pd.Timestamp(entry_dt)
    f = history_features(prior)
    f["target_dist_pct"] = (take_profit - entry_price) / entry_price
    f["entry_hour"], f["entry_dow"], f["entry_month"] = entry_dt.hour, entry_dt.dayofweek, entry_dt.month
    if "_last_exit_price" in f:
        last_px = f.pop("_last_exit_price")
        f["gap_pct"] = (entry_price - last_px) / last_px
        f["days_since_last_exit"] = (entry_dt - f.pop("_last_exit_dt")).total_seconds() / 86400
    else:
        f["gap_pct"] = f["days_since_last_exit"] = np.nan
    return {k: f[k] for k in FEATURES}


def prior_history(trades: pd.DataFrame, entry_dt, mode: str = "t1") -> pd.DataFrame:
    """Closed trades visible at `entry_dt` under the given timing mode."""
    entry_dt = pd.Timestamp(entry_dt)
    if mode == "t1":
        mask = trades["exit_dt"] < entry_dt.normalize()
    elif mode == "sequential":
        mask = trades["exit_dt"] <= entry_dt
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return trades[mask].sort_values("exit_dt")


def build_feature_table(trades: pd.DataFrame, mode: str = "t1") -> pd.DataFrame:
    """One feature row per trade. Rows without enough history are dropped."""
    trades = trades.sort_values("entry_dt").reset_index(drop=True)
    rows = []
    for t in trades.itertuples(index=False):
        prior = prior_history(trades, t.entry_dt, mode)
        rows.append(entry_features(t.entry_dt, t.entry_price, t.take_profit, prior))
    X = pd.DataFrame(rows, columns=FEATURES).replace([np.inf, -np.inf], np.nan)
    out = pd.concat([trades, X], axis=1)
    out["win"] = (out["pnl"] > 0).astype(int)
    return out.dropna(subset=FEATURES).reset_index(drop=True)
