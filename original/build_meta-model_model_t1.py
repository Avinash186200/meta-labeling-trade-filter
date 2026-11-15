"""
Stricter T-1 robustness variant of the META-MODEL meta-labeling model.

The original rolling features (roll_winrate_N, roll_avgpnl_N, streak,
prev_trade_pnl, prev_duration_min, gap_pct, days_since_last_exit) used the
immediately-PRECEDING TRADE in sequence. 38.2% of trades have that prior
trade closing on the SAME calendar day as the current trade's entry --
technically same-day information, not strictly "known before today's open."

This script rebuilds those same features using only trades that CLOSED
strictly before the current trade's entry date (a true T-1 / prior-close
cutoff, with zero same-day information), and re-runs the identical
walk-forward test to see whether the edge survives the stricter cut.

target_dist_pct is unchanged: it is a property of the trade being scored
itself (its own Take_Profit vs its own Entry_Price), known the instant that
trade is placed -- this is not a T-1 issue, it's the whole point of
meta-labeling (scoring a signal using that signal's own attributes).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 160)
START_EQUITY = 50000.0

df = pd.read_csv(
    "<local-path>",
    parse_dates=["entry_dt", "entry_date", "exit_date"],
).sort_values("entry_dt").reset_index(drop=True)

# ---------------------------------------------------------------------------
# T-1 anchor: for trade i, find j = last trade whose exit_date < entry_date[i]
# ---------------------------------------------------------------------------
exit_dates = df["exit_date"].values
entry_dates = df["entry_date"].values
# searchsorted needs exit_dates sorted -- true here since one position open at a time
anchor_idx = np.searchsorted(exit_dates, entry_dates, side="left") - 1  # last idx with exit_date < entry_date[i]

win = df["win"].values.astype(float)
pnl = df["pnl"].values.astype(float)

n = len(df)
roll_winrate_5 = np.full(n, np.nan); roll_avgpnl_5 = np.full(n, np.nan)
roll_winrate_10 = np.full(n, np.nan); roll_avgpnl_10 = np.full(n, np.nan)
roll_winrate_20 = np.full(n, np.nan); roll_avgpnl_20 = np.full(n, np.nan)
roll_vol_10 = np.full(n, np.nan)
prev_trade_pnl_t1 = np.full(n, np.nan)
streak_t1 = np.zeros(n)
days_since_last_exit_t1 = np.full(n, np.nan)

for i in range(n):
    j = anchor_idx[i]  # inclusive last usable prior-trade index (strictly earlier calendar day)
    if j < 0:
        continue
    prev_trade_pnl_t1[i] = pnl[j]
    days_since_last_exit_t1[i] = (df["entry_dt"].iloc[i] - df["exit_date"].iloc[j]).total_seconds() / 86400.0
    for N, wr_arr, pnl_arr in [(5, roll_winrate_5, roll_avgpnl_5), (10, roll_winrate_10, roll_avgpnl_10), (20, roll_winrate_20, roll_avgpnl_20)]:
        lo = max(0, j - N + 1)
        window = win[lo:j+1]
        if len(window) >= max(3, N // 2):
            wr_arr[i] = window.mean()
            pnl_arr[i] = pnl[lo:j+1].mean()
    lo10 = max(0, j - 9)
    if j - lo10 + 1 >= 5:
        roll_vol_10[i] = pnl[lo10:j+1].std()
    # streak as of end of day (T-1): walk backwards from j
    s = 0
    k = j
    if k >= 0:
        first = win[k]
        s = 1 if first == 1 else -1
        k -= 1
        while k >= 0 and win[k] == first:
            s += 1 if first == 1 else -1
            k -= 1
    streak_t1[i] = s

df["roll_winrate_5"] = roll_winrate_5; df["roll_avgpnl_5"] = roll_avgpnl_5
df["roll_winrate_10"] = roll_winrate_10; df["roll_avgpnl_10"] = roll_avgpnl_10
df["roll_winrate_20"] = roll_winrate_20; df["roll_avgpnl_20"] = roll_avgpnl_20
df["roll_vol_10"] = roll_vol_10
df["prev_trade_pnl"] = prev_trade_pnl_t1
df["streak"] = streak_t1
df["days_since_last_exit"] = days_since_last_exit_t1
# gap_pct / prev_duration_min need the T-1 anchor's exit price / duration too
if "exit_price" in df.columns:
    prev_exit_price_t1 = np.full(n, np.nan)
    for i in range(n):
        j = anchor_idx[i]
        if j >= 0:
            prev_exit_price_t1[i] = df["exit_price"].iloc[j]
    df["gap_pct"] = (df["entry_price"] - prev_exit_price_t1) / prev_exit_price_t1
if "entry_dt" in df.columns:
    prev_duration_t1 = np.full(n, np.nan)
    for i in range(n):
        j = anchor_idx[i]
        if j >= 0:
            prev_duration_t1[i] = (df["exit_date"].iloc[j] - df["entry_dt"].iloc[j]).total_seconds() / 60.0
    df["prev_duration_min"] = prev_duration_t1

FEATURES = [
    "target_dist_pct",
    "entry_hour", "entry_dow", "entry_month",
    "gap_pct", "days_since_last_exit", "prev_trade_pnl", "prev_duration_min",
    "roll_winrate_5", "roll_avgpnl_5",
    "roll_winrate_10", "roll_avgpnl_10",
    "roll_winrate_20", "roll_avgpnl_20",
    "roll_vol_10", "streak",
]
df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
model_df = df.dropna(subset=FEATURES + ["win"]).reset_index(drop=True)
print(f"T-1 strict variant: {len(model_df)} usable trades (dropped {len(df)-len(model_df)} warm-up)")

years = sorted(model_df["entry_year"].unique())
test_years = years[2:]

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
    scaler = StandardScaler().fit(model_df.loc[train_mask, FEATURES].values)
    X_train = scaler.transform(model_df.loc[train_mask, FEATURES].values)
    X_test = scaler.transform(model_df.loc[test_mask, FEATURES].values)
    y_train = model_df.loc[train_mask, "win"].values
    for name, factory in MODELS.items():
        clf = factory().fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        idx = model_df.index[test_mask]
        rec = model_df.loc[idx, ["entry_dt", "entry_date", "exit_date", "pnl", "win"]].copy()
        rec["proba"] = proba
        oos_records[name].append(rec)

oos = {name: pd.concat(recs, ignore_index=True) for name, recs in oos_records.items() if recs}


def daily_equity(pnl_df):
    daily = pnl_df.groupby("exit_date")["pnl"].sum().sort_index()
    days = pd.bdate_range(daily.index.min(), daily.index.max())
    daily = daily.reindex(days, fill_value=0.0)
    equity = START_EQUITY + daily.cumsum()
    ret = daily / equity.shift(1).fillna(START_EQUITY)
    return daily, equity, ret


def stats(pnl_df):
    daily, equity, ret = daily_equity(pnl_df)
    ny = (daily.index[-1] - daily.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / START_EQUITY) ** (1 / ny) - 1
    sharpe = ret.mean() / ret.std() * np.sqrt(252)
    mdd = ((equity - equity.cummax()) / equity.cummax()).min()
    return dict(n=len(pnl_df), win_rate=(pnl_df.pnl > 0).mean(), cagr=cagr, sharpe=sharpe, max_dd=mdd, final=equity.iloc[-1])


print("\n" + "=" * 100)
print("T-1 STRICT (prior-calendar-day-only) FEATURES -- WALK-FORWARD OOS")
print("=" * 100)
KEEP_FRAC = 0.75  # canonical cut used in the deck: keep top 75%, drop weakest quarter
rows = []
for name, rec in oos.items():
    base = stats(rec)
    for keep in (0.90, 0.75, 0.55, 0.50):
        cutoff = rec["proba"].quantile(1 - keep)
        filt = rec[rec["proba"] >= cutoff]
        f = stats(filt)
        print(f"{name:12s} keep~{keep:.2f}  base sharpe={base['sharpe']:.2f} wr={base['win_rate']:.3f}  "
              f"| filtered n={f['n']:4d} sharpe={f['sharpe']:.2f} cagr={f['cagr']*100:5.1f}% "
              f"mdd={f['max_dd']*100:6.1f}% wr={f['win_rate']:.3f}")
        if keep == KEEP_FRAC:
            rows.append({"model": name, "label": "Baseline (all OOS trades)", **base})
            rows.append({"model": name, "label": f"META-MODEL-filtered (keep top {int(keep*100)}%)",
                         **f, "pct_trades_kept": len(filt) / len(rec)})
            daily, equity, ret = daily_equity(rec)
            equity.to_csv(rf"<local-path>", header=["equity"])
            daily_f, equity_f, ret_f = daily_equity(filt)
            equity_f.to_csv(rf"<local-path>", header=["equity"])

pd.DataFrame(rows).to_csv("<local-path>", index=False)
model_df.to_csv("<local-path>", index=False)
print(f"\nSaved output/meta-model_oos_performance.csv (canonical, T-1 strict, keep={KEEP_FRAC}), "
      "output/equity_*.csv, output/model_features_t1_strict.csv")
