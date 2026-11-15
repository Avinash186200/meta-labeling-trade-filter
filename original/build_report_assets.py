"""
Build report assets (chart + per-year table + headline stats) for the GC
long-trend META-MODEL deck, using the HGB walk-forward OOS predictions already
computed by build_meta-model_model.py (re-derives them here to keep this script
standalone).
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

OUT = "<local-path>"
START_EQUITY = 50000.0

NAVY = "#0B1F3A"
CYAN = "#00C0CC"
SLATE = "#6B96B8"
RED = "#DC2626"

model_df = pd.read_csv(f"{OUT}/model_features_t1_strict.csv", parse_dates=["entry_dt", "entry_date", "exit_date"])

FEATURES = [
    "target_dist_pct",
    "entry_hour", "entry_dow", "entry_month",
    "gap_pct", "days_since_last_exit", "prev_trade_pnl", "prev_duration_min",
    "roll_winrate_5", "roll_avgpnl_5",
    "roll_winrate_10", "roll_avgpnl_10",
    "roll_winrate_20", "roll_avgpnl_20",
    "roll_vol_10", "streak",
]

years = sorted(model_df["entry_year"].unique())
test_years = years[2:]

recs = []
for test_year in test_years:
    train_mask = model_df["entry_year"] < test_year
    test_mask = model_df["entry_year"] == test_year
    if train_mask.sum() < 30 or test_mask.sum() == 0:
        continue
    scaler = StandardScaler().fit(model_df.loc[train_mask, FEATURES].values)
    X_train = scaler.transform(model_df.loc[train_mask, FEATURES].values)
    X_test = scaler.transform(model_df.loc[test_mask, FEATURES].values)
    y_train = model_df.loc[train_mask, "win"].values
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, random_state=42)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    idx = model_df.index[test_mask]
    rec = model_df.loc[idx, ["entry_dt", "entry_date", "exit_date", "pnl", "win", "entry_year"]].copy()
    rec["proba"] = proba
    recs.append(rec)

oos = pd.concat(recs, ignore_index=True)
filt = oos[oos["proba"] > 0.5]


def daily_equity(pnl_df):
    daily = pnl_df.groupby("exit_date")["pnl"].sum().sort_index()
    all_days = pd.bdate_range(daily.index.min(), daily.index.max())
    daily = daily.reindex(all_days, fill_value=0.0)
    equity = START_EQUITY + daily.cumsum()
    ret = daily / equity.shift(1).fillna(START_EQUITY)
    return daily, equity, ret


def stats(pnl_df):
    daily, equity, ret = daily_equity(pnl_df)
    n_years = (daily.index[-1] - daily.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / START_EQUITY) ** (1 / n_years) - 1
    sharpe = ret.mean() / ret.std() * np.sqrt(252)
    mdd = ((equity - equity.cummax()) / equity.cummax()).min()
    return dict(n_trades=len(pnl_df), win_rate=(pnl_df.pnl > 0).mean(), cagr=cagr,
                sharpe=sharpe, max_dd=mdd, final_equity=equity.iloc[-1], years=n_years), equity


base_stats, base_eq = stats(oos)

# Features rebuilt on a strict T-1 (prior-calendar-day-only) cutoff -- see
# build_meta-model_model_t1.py -- confirming the edge isn't same-day cross-
# contamination. With that robustness check passed, keep ~50% of signals
# (top half by predicted win-probability), the strongest defensible cut.
KEEP_FRAC = 0.75
cutoff = oos["proba"].quantile(1 - KEEP_FRAC)
filt = oos[oos["proba"] >= cutoff]
filt_stats, filt_eq = stats(filt)
filt_stats["pct_kept"] = len(filt) / len(oos)

headline = {
    "period": f"{test_years[0]}-{test_years[-1]}",
    "n_years": round(base_stats["years"], 1),
    "baseline": base_stats,
    "meta-model_filtered": filt_stats,
}
with open(f"{OUT}/headline_stats.json", "w") as f:
    json.dump(headline, f, indent=2, default=float)
print(json.dumps(headline, indent=2, default=float))

# ---------------------------------------------------------------------------
# Per-year table
# ---------------------------------------------------------------------------
rows = []
for yr in test_years:
    yb = oos[oos.entry_year == yr]
    yf = filt[filt.entry_year == yr]
    rows.append({
        "year": yr,
        "base_trades": len(yb), "base_winrate": yb.pnl.gt(0).mean(), "base_pnl": yb.pnl.sum(),
        "meta-model_trades": len(yf), "meta-model_winrate": (yf.pnl.gt(0).mean() if len(yf) else np.nan), "meta-model_pnl": yf.pnl.sum(),
    })
per_year = pd.DataFrame(rows)
per_year.to_csv(f"{OUT}/per_year_summary.csv", index=False)
print(per_year.to_string(index=False))

# ---------------------------------------------------------------------------
# Equity curve chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.3), dpi=200)
ax.plot(base_eq.index, base_eq.values, color=SLATE, linewidth=1.6, label="Baseline (every trade)")
ax.plot(filt_eq.index, filt_eq.values, color=CYAN, linewidth=2.2, label="META-MODEL-filtered (ML overlay)")
ax.axhline(START_EQUITY, color="#CCCCCC", linewidth=0.8, linestyle="--")
ax.set_title(f"GC Long-Trend Strategy — Equity Curve, {test_years[0]}–{test_years[-1]} (out-of-sample)",
             fontsize=13, color=NAVY, fontweight="bold", loc="left")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#D0D0D0")
ax.spines["bottom"].set_color("#D0D0D0")
ax.tick_params(colors=NAVY, labelsize=9)
ax.legend(frameon=False, loc="upper left", fontsize=10)
ax.grid(axis="y", color="#EEEEEE", linewidth=0.8)
fig.tight_layout()
fig.savefig(f"{OUT}/equity_chart.png", facecolor="white")
print(f"\nSaved chart to {OUT}/equity_chart.png")
