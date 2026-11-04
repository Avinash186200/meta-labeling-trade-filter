"""Portfolio statistics from a list of closed trades."""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_equity(trades: pd.DataFrame, start_equity: float,
                 index: pd.DatetimeIndex | None = None) -> tuple[pd.Series, pd.Series]:
    """Business-day equity curve and simple returns, P&L booked on exit date."""
    daily = trades.groupby(pd.to_datetime(trades["exit_date"]).dt.normalize())["pnl"].sum()
    if index is None:
        index = pd.bdate_range(daily.index.min(), daily.index.max())
    daily = daily.reindex(index, fill_value=0.0)
    equity = start_equity + daily.cumsum()
    ret = daily / equity.shift(1).fillna(start_equity)
    return equity, ret


def perf_stats(trades: pd.DataFrame, start_equity: float = 50_000.0,
               index: pd.DatetimeIndex | None = None) -> dict:
    """Sharpe (daily, sqrt-252), CAGR, max drawdown, win rate.

    Pass the same `index` for a baseline and a filtered run so both are
    measured over the same calendar (otherwise skipped days at the edges
    shorten the filtered period and flatter it).
    """
    equity, ret = daily_equity(trades, start_equity, index)
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    final = equity.iloc[-1]
    return {
        "n_trades": int(len(trades)),
        "win_rate": float((trades["pnl"] > 0).mean()) if len(trades) else np.nan,
        "total_return": float(final / start_equity - 1),
        "cagr": float((final / start_equity) ** (1 / years) - 1) if years > 0 and final > 0 else np.nan,
        "sharpe": float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else np.nan,
        "max_drawdown": float(((equity - equity.cummax()) / equity.cummax()).min()),
    }


def compare(oos: pd.DataFrame, start_equity: float = 50_000.0) -> pd.DataFrame:
    """Baseline (take all) vs filtered (keep only) stats on one shared calendar."""
    idx = pd.bdate_range(pd.to_datetime(oos["exit_date"]).min(), pd.to_datetime(oos["exit_date"]).max())
    base = perf_stats(oos, start_equity, idx)
    filt = perf_stats(oos[oos["keep"]], start_equity, idx)
    filt["pct_kept"] = float(oos["keep"].mean())
    base["pct_kept"] = 1.0
    return pd.DataFrame([base, filt], index=["baseline", "filtered"])
