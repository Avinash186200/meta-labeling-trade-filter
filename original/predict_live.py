"""
Reference implementation of the GC model's live scoring logic -- this is
what a `/score` API endpoint would run. Not a server; just the scoring
function, so the actual API wrapper (Flask/FastAPI) has something concrete
to call.

Required from the trader, per new trade (at entry):
    entry_price       float   -- fill price
    take_profit       float   -- REQUIRED. Set at entry, same as always.
    entry_dt          datetime

Required from stored history (server-side, built from the trader's ingested
closed trades -- NOT re-entered by the trader each time):
    a list of prior CLOSED trades, each with exit_date, exit_dt, pnl,
    entry_dt/exit_dt duration -- i.e. exactly what a daily trade/PnL upload
    already contains.

Everything else (rolling win-rate, streak, gaps, etc.) is derived here.
"""
import joblib
import numpy as np
import pandas as pd

OUT = "<local-path>"
BUNDLE = joblib.load(f"{OUT}/gc_production_model.joblib")
MODEL, SCALER, FEATURES, CUTOFF = BUNDLE["model"], BUNDLE["scaler"], BUNDLE["features"], BUNDLE["probability_cutoff"]


def score_new_trade(entry_price, take_profit, entry_dt, history_df):
    """
    history_df: DataFrame of prior CLOSED trades for this same strategy,
    columns [exit_date, exit_dt, exit_price, pnl, entry_dt] (entry_dt = when
    that prior trade was opened, used for its own duration). Must be sorted
    ascending by exit_dt; the T-1 filter below drops any same-day trades.
    """
    entry_date = pd.Timestamp(entry_dt).normalize()
    prior = history_df[history_df["exit_date"] < entry_date]  # T-1 strict: no same-day info
    if prior.empty:
        raise ValueError("No prior-day trade history available -- model needs warm-up history first.")

    last = prior.iloc[-1]
    win = (prior["pnl"] > 0).astype(int)

    def roll(n, col_fn):
        window = prior.tail(n)
        return col_fn(window) if len(window) >= max(3, n // 2) else np.nan

    feats = {
        "target_dist_pct": (take_profit - entry_price) / entry_price,
        "entry_hour": pd.Timestamp(entry_dt).hour,
        "entry_dow": pd.Timestamp(entry_dt).dayofweek,
        "entry_month": pd.Timestamp(entry_dt).month,
        "gap_pct": (entry_price - last["exit_price"]) / last["exit_price"] if "exit_price" in prior.columns else np.nan,
        "days_since_last_exit": (pd.Timestamp(entry_dt) - last["exit_date"]).total_seconds() / 86400.0,
        "prev_trade_pnl": last["pnl"],
        "prev_duration_min": (last["exit_dt"] - last["entry_dt"]).total_seconds() / 60.0,
        "roll_winrate_5": roll(5, lambda w: (w["pnl"] > 0).mean()),
        "roll_avgpnl_5": roll(5, lambda w: w["pnl"].mean()),
        "roll_winrate_10": roll(10, lambda w: (w["pnl"] > 0).mean()),
        "roll_avgpnl_10": roll(10, lambda w: w["pnl"].mean()),
        "roll_winrate_20": roll(20, lambda w: (w["pnl"] > 0).mean()),
        "roll_avgpnl_20": roll(20, lambda w: w["pnl"].mean()),
        "roll_vol_10": roll(10, lambda w: w["pnl"].std()),
        "streak": _streak(win.values),
    }

    x = np.array([[feats[f] for f in FEATURES]])
    if np.isnan(x).any():
        missing = [f for f, v in zip(FEATURES, x[0]) if np.isnan(v)]
        raise ValueError(f"Not enough history to compute: {missing}")

    xs = SCALER.transform(x)
    proba = MODEL.predict_proba(xs)[0, 1]
    return {
        "win_probability": float(proba),
        "take_trade": bool(proba >= CUTOFF),
        "cutoff_used": CUTOFF,
    }


def _streak(win_array):
    if len(win_array) == 0:
        return 0
    s = 0
    first = win_array[-1]
    for v in win_array[::-1]:
        if v == first:
            s += 1 if first == 1 else -1
        else:
            break
    return s


if __name__ == "__main__":
    print("Loaded production model. Feature order:", FEATURES)
    print("Decision cutoff (P(win) >=):", CUTOFF)
    print("This module exposes score_new_trade(entry_price, take_profit, entry_dt, history_df) "
          "for the API layer to call.")
