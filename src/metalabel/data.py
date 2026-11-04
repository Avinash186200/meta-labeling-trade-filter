"""Trade-log loading and synthetic data generation.

Input format: a broker-style fill log with one Entry row and one Exit row per
Position_ID. `pair_trades` turns it into one row per closed position.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PAIRED_COLUMNS = [
    "position_id", "entry_dt", "entry_price", "take_profit", "stop_loss",
    "exit_dt", "exit_date", "exit_price", "pnl", "exit_reason",
]


def pair_trades(raw: pd.DataFrame) -> pd.DataFrame:
    """Pair Entry/Exit fill rows by Position_ID into one row per closed trade."""
    df = raw.copy()
    df["Trade_Date"] = pd.to_datetime(df["Trade_Date"])
    df["ts"] = pd.to_datetime(df["Trade_Date"].dt.strftime("%Y-%m-%d") + " " + df["Trade_Time"])
    entries = df[df["Entry_Exit"] == "Entry"].set_index("Position_ID")
    exits = df[df["Entry_Exit"] == "Exit"].set_index("Position_ID")
    paired = pd.DataFrame({
        "entry_dt": entries["ts"],
        "entry_price": entries["Price"],
        "take_profit": entries["Take_Profit"],
        "stop_loss": entries["Stop_Loss"],
        "exit_dt": exits["ts"],
        "exit_date": exits["Trade_Date"],
        "exit_price": exits["Price"],
        "pnl": exits["Realized_PnL"],
        "exit_reason": exits["Notes"],
    }).dropna(subset=["pnl"])  # still-open positions have no exit row
    paired.index.name = "position_id"
    return paired.reset_index().sort_values("entry_dt").reset_index(drop=True)[PAIRED_COLUMNS]


def load_trade_log(path) -> pd.DataFrame:
    """Read a fill-log CSV and return one row per closed trade."""
    return pair_trades(pd.read_csv(path))


def merge_incremental(existing: pd.DataFrame, new_paired: pd.DataFrame) -> pd.DataFrame:
    """Idempotent merge: re-sent positions replace earlier copies."""
    out = pd.concat([existing, new_paired], ignore_index=True)
    out = out.drop_duplicates(subset=["position_id"], keep="last")
    return out.sort_values("entry_dt").reset_index(drop=True)


def make_synthetic_log(n_trades: int = 1500, start: str = "2016-01-04", seed: int = 7,
                       signal_strength: float = 1.0) -> pd.DataFrame:
    """Generate a fill log with the same columns as a real export.

    A long-only trend strategy on a generic futures contract. Win probability
    depends weakly on take-profit distance and on a slow regime variable, so a
    meta-model has something real (but modest) to find. `signal_strength=0`
    gives a pure-noise log, which the tests use as a leakage check.
    """
    rng = np.random.default_rng(seed)
    day = pd.Timestamp(start)
    price = 1200.0
    regime = 0.0
    rows = []
    for i in range(n_trades):
        day += pd.offsets.BDay(int(rng.integers(0, 3)))
        regime = 0.97 * regime + rng.normal(0, 0.3)
        price *= float(np.exp(rng.normal(0.0002, 0.01)))
        hour, minute = int(rng.integers(8, 15)), int(rng.choice([0, 15, 30, 45]))
        entry_ts = day + pd.Timedelta(hours=hour, minutes=minute)
        tp_dist = float(rng.uniform(0.003, 0.03))
        logit = signal_strength * (-60 * (tp_dist - 0.015) + 0.8 * regime) - 0.3
        win = rng.random() < 1 / (1 + np.exp(-logit))
        qty = 1
        tp = price * (1 + tp_dist)
        if win:
            exit_px, reason = tp, "take_profit"
        else:
            exit_px, reason = price * (1 - float(rng.uniform(0.008, 0.022))), "stop_loss"
        hold = pd.Timedelta(minutes=int(rng.integers(20, 60 * 30)))
        exit_ts = entry_ts + hold
        pnl = (exit_px - price) * qty * 10  # contract multiplier 10
        pid = f"P{i:06d}"
        base = dict(Account_ID="ACC-DEMO", Strategy_ID="trend_long", Timezone="UTC",
                    Symbol="FUT1", Asset_Class="Future", Exchange="EXCH", Currency="USD",
                    Order_Type="MKT", Quantity=qty, Position_ID=pid,
                    Stop_Loss=round(price, 2), Take_Profit=round(tp, 2))
        rows.append({**base, "Trade_ID": f"T{2*i:07d}", "Trade_Date": entry_ts.date(),
                     "Trade_Time": entry_ts.strftime("%H:%M:%S"), "Side": "BUY",
                     "Price": round(price, 2), "Entry_Exit": "Entry", "Realized_PnL": np.nan,
                     "Notes": ""})
        rows.append({**base, "Trade_ID": f"T{2*i+1:07d}", "Trade_Date": exit_ts.date(),
                     "Trade_Time": exit_ts.strftime("%H:%M:%S"), "Side": "SELL",
                     "Price": round(exit_px, 2), "Entry_Exit": "Exit",
                     "Realized_PnL": round(pnl, 2), "Notes": reason})
        price = exit_px
        day = exit_ts.normalize()
    return pd.DataFrame(rows)
