"""
Meta-model (META-MODEL) meta-labeling overlay for the GC long-trend strategy.

Base strategy: long-only GC (gold futures) trend strategy, one trade at a time,
fixed stop-loss / take-profit per position (see trades CSV).

Approach (meta-labeling, since no independent market data feed lines up with
this specific trade log's price level -> features are derived from the trade
log itself: entry structure, recent trade history, timing/seasonality):

  1. Build one row per trade (at entry time) using only information known
     at/ before entry: stop/target structure, time-of-day/day-of-week/month,
     gap since prior exit, rolling win-rate/PnL/streak/vol of PRIOR trades.
  2. Target = 1 if that trade's Realized_PnL > 0 else 0.
  3. Walk-forward (expanding window, year-by-year): train on all trades up to
     year Y, predict year Y+1. First 2 years are warm-up (train only, no OOS
     prediction) since there isn't enough history yet.
  4. META-MODEL filter = take the trade only if predicted P(win) > 0.5 (standard
     meta-label cutoff, no threshold-fitting on test data).
  5. Compare baseline (take every trade) vs META-MODEL-filtered equity curve:
     Sharpe, CAGR, max drawdown, win rate, % of trades kept.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 160)

TRADES_PATH = "<local-path>"
START_EQUITY = 50000.0

# ---------------------------------------------------------------------------
# 1. Load and pair entries with exits (one row per closed position)
# ---------------------------------------------------------------------------
df = pd.read_csv(TRADES_PATH, parse_dates=["Trade_Date"])
df["entry_dt"] = pd.to_datetime(df["Trade_Date"].dt.date.astype(str) + " " + df["Trade_Time"])

entries = df[df.Entry_Exit == "Entry"].set_index("Position_ID")
exits = df[df.Entry_Exit == "Exit"].set_index("Position_ID")

trades = pd.DataFrame({
    "entry_dt": entries["entry_dt"],
    "entry_date": entries["Trade_Date"],
    "entry_price": entries["Avg_Entry_Price"],
    "stop_loss": entries["Stop_Loss"],
    "take_profit": entries["Take_Profit"],
    "exit_dt": exits["entry_dt"],
    "exit_date": exits["Trade_Date"],
    "exit_price": exits["Price"],
    "pnl": exits["Realized_PnL"],
    "exit_reason": exits["Notes"],
}).dropna(subset=["pnl"]).sort_values("entry_dt").reset_index(drop=True)

print(f"Paired trades: {len(trades)}")
assert trades["pnl"].notna().all()

# ---------------------------------------------------------------------------
# 2. Feature engineering (entry-time only, no look-ahead)
# ---------------------------------------------------------------------------
trades["win"] = (trades["pnl"] > 0).astype(int)

trades["target_dist_pct"] = (trades["take_profit"] - trades["entry_price"]) / trades["entry_price"]
# note: Stop_Loss == Avg_Entry_Price at entry time for every trade in this log
# (initial stop is set at breakeven / tightened by a trailing mechanism not
# captured in the entry row), so it carries no cross-sectional signal and is
# dropped rather than engineered into a meaningless ratio.

trades["entry_hour"] = trades["entry_dt"].dt.hour
trades["entry_dow"] = trades["entry_dt"].dt.dayofweek
trades["entry_month"] = trades["entry_dt"].dt.month
trades["entry_year"] = trades["entry_dt"].dt.year

trades["prev_exit_price"] = trades["exit_price"].shift(1)
trades["prev_exit_dt"] = trades["exit_dt"].shift(1)
trades["gap_pct"] = (trades["entry_price"] - trades["prev_exit_price"]) / trades["prev_exit_price"]
trades["days_since_last_exit"] = (trades["entry_dt"] - trades["prev_exit_dt"]).dt.total_seconds() / 86400.0
trades["prev_trade_pnl"] = trades["pnl"].shift(1)
trades["prev_duration_min"] = (trades["exit_dt"].shift(1) - trades["entry_dt"].shift(1)).dt.total_seconds() / 60.0

# rolling stats over PRIOR trades only (shift(1) before rolling so trade i doesn't see its own outcome)
prior_win = trades["win"].shift(1)
prior_pnl = trades["pnl"].shift(1)
for n in (5, 10, 20):
    trades[f"roll_winrate_{n}"] = prior_win.rolling(n, min_periods=max(3, n // 2)).mean()
    trades[f"roll_avgpnl_{n}"] = prior_pnl.rolling(n, min_periods=max(3, n // 2)).mean()
trades["roll_vol_10"] = prior_pnl.rolling(10, min_periods=5).std()

# current streak going into this trade (positive = win streak, negative = loss streak)
streak = np.zeros(len(trades))
s = 0
prev_wins = trades["win"].shift(1).values
for i in range(len(trades)):
    if i == 0 or np.isnan(prev_wins[i]):
        s = 0
    else:
        if prev_wins[i] == 1:
            s = s + 1 if s >= 0 else 1
        else:
            s = s - 1 if s <= 0 else -1
    streak[i] = s
trades["streak"] = streak

FEATURES = [
    "target_dist_pct",
    "entry_hour", "entry_dow", "entry_month",
    "gap_pct", "days_since_last_exit", "prev_trade_pnl", "prev_duration_min",
    "roll_winrate_5", "roll_avgpnl_5",
    "roll_winrate_10", "roll_avgpnl_10",
    "roll_winrate_20", "roll_avgpnl_20",
    "roll_vol_10", "streak",
]

trades[FEATURES] = trades[FEATURES].replace([np.inf, -np.inf], np.nan)
model_df = trades.dropna(subset=FEATURES + ["win"]).reset_index(drop=True)
print(f"Usable trades after feature warm-up: {len(model_df)} (dropped {len(trades)-len(model_df)} warm-up rows)")

# ---------------------------------------------------------------------------
# 3. Walk-forward, expanding window, year by year
# ---------------------------------------------------------------------------
years = sorted(model_df["entry_year"].unique())
warmup_years = years[:2]           # first 2 years -> train-only warm-up
test_years = years[2:]
print("years:", years, " warmup:", warmup_years, " test:", test_years)

MODELS = {
    "LogReg": lambda: LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=20, random_state=42),
    "HGB": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, random_state=42),
}

oos_records = {name: [] for name in MODELS}

for test_year in test_years:
    train_mask = model_df["entry_year"] < test_year
    test_mask = model_df["entry_year"] == test_year
    if train_mask.sum() < 30 or test_mask.sum() == 0:
        continue
    X_train_raw = model_df.loc[train_mask, FEATURES].values
    y_train = model_df.loc[train_mask, "win"].values
    X_test_raw = model_df.loc[test_mask, FEATURES].values

    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    for name, factory in MODELS.items():
        clf = factory()
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        idx = model_df.index[test_mask]
        rec = model_df.loc[idx, ["entry_dt", "entry_date", "exit_date", "pnl", "win"]].copy()
        rec["proba"] = proba
        rec["model"] = name
        oos_records[name].append(rec)

oos = {name: pd.concat(recs, ignore_index=True) for name, recs in oos_records.items() if recs}
for name, rec in oos.items():
    print(f"\n{name}: {len(rec)} OOS trades, years {test_years[0]}-{test_years[-1]}")

# ---------------------------------------------------------------------------
# 4. Performance: baseline vs META-MODEL-filtered (per model), on the OOS window only
# ---------------------------------------------------------------------------
def daily_equity_from_trades(pnl_df, date_col="exit_date"):
    daily = pnl_df.groupby(date_col)["pnl"].sum().sort_index()
    all_days = pd.bdate_range(daily.index.min(), daily.index.max())
    daily = daily.reindex(all_days, fill_value=0.0)
    equity = START_EQUITY + daily.cumsum()
    ret = daily / equity.shift(1).fillna(START_EQUITY)
    return daily, equity, ret


def perf_stats(pnl_df, date_col="exit_date", label=""):
    daily, equity, ret = daily_equity_from_trades(pnl_df, date_col)
    n_years = (daily.index[-1] - daily.index[0]).days / 365.25
    total_ret = equity.iloc[-1] / START_EQUITY - 1
    cagr = (equity.iloc[-1] / START_EQUITY) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan
    roll_max = equity.cummax()
    mdd = ((equity - roll_max) / roll_max).min()
    win_rate = (pnl_df["pnl"] > 0).mean()
    return {
        "label": label,
        "n_trades": len(pnl_df),
        "win_rate": win_rate,
        "total_return": total_ret,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": mdd,
        "final_equity": equity.iloc[-1],
        "years": n_years,
    }, equity


results = []
equity_curves = {}

KEEP_FRAC = 0.75  # keep top 75% by predicted win-probability, drop the weakest quarter

for name, rec in oos.items():
    base_stats, base_eq = perf_stats(rec, label=f"{name} | Baseline (all OOS trades)")
    cutoff = rec["proba"].quantile(1 - KEEP_FRAC)
    filt = rec[rec["proba"] >= cutoff]
    if len(filt) < 10:
        continue
    filt_stats, filt_eq = perf_stats(filt, label=f"{name} | META-MODEL-filtered (bottom decile removed)")
    filt_stats["pct_trades_kept"] = len(filt) / len(rec)
    results.append(base_stats)
    results.append(filt_stats)
    equity_curves[f"{name}_baseline"] = base_eq
    equity_curves[f"{name}_filtered"] = filt_eq

res_df = pd.DataFrame(results)
print("\n" + "=" * 100)
print("OOS PERFORMANCE, BASELINE VS META-MODEL-FILTERED (walk-forward, expanding window)")
print("=" * 100)
print(res_df.to_string(index=False))

res_df.to_csv("<local-path>", index=False)
for k, v in equity_curves.items():
    v.to_csv(rf"<local-path>", header=["equity"])

model_df.to_csv("<local-path>", index=False)
print("\nSaved: output/meta-model_oos_performance.csv, output/equity_*.csv, output/model_features.csv")
