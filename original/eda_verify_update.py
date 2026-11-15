import pandas as pd
import numpy as np

old_path = "<local-path>"
new_path = "<local-path>"
eq_path  = "<local-path>"

old = pd.read_csv(old_path, parse_dates=["Trade_Date"])
new = pd.read_csv(new_path, parse_dates=["Trade_Date"])
eq  = pd.read_csv(eq_path, parse_dates=["Date"])

print("="*80, "\nSHAPE CHECK\n", "="*80)
print("old:", old.shape, "new:", new.shape, "equity:", eq.shape)

print("\n" + "="*80, "\nIS REALIZED_PNL IDENTICAL OLD VS NEW (per Trade_ID)?\n", "="*80)
m = old[["Trade_ID","Realized_PnL"]].merge(new[["Trade_ID","Realized_PnL"]], on="Trade_ID", suffixes=("_old","_new"))
diff = m[(m.Realized_PnL_old.fillna(0) - m.Realized_PnL_new.fillna(0)).abs() > 1e-6]
print("rows with differing PnL:", len(diff), "/", len(m))
print(diff.head(10))

print("\n" + "="*80, "\nCOMMISSION FIELD CHECK\n", "="*80)
print(new.Commission.describe())
print(new.Fees.describe())
print("commission unique:", new.Commission.unique())
print("fees unique:", new.Fees.unique())

print("\n" + "="*80, "\nTOTALS\n", "="*80)
exits_new = new[new.Entry_Exit=="Exit"]
total_realized_pnl = exits_new.Realized_PnL.sum()
n_trades = len(exits_new)
total_commission_legs = new.Commission.sum()  # commission charged per leg (entry+exit)
print("n round-trip trades:", n_trades)
print("sum Realized_PnL (new file):", total_realized_pnl)
print("sum Commission across ALL legs (entry+exit):", total_commission_legs)
print("implied per-trade roundtrip commission ($2.5 entry + $2.5 exit):", new.Commission.iloc[0]*2)

print("\n" + "="*80, "\nDOES EQUITY FILE'S CUMULATIVE PNL MATCH SUM OF TRADE REALIZED_PNL?\n", "="*80)
final_cum_pnl = eq.Cumulative_PnL.iloc[-1]
print("equity file final Cumulative_PnL:", final_cum_pnl)
print("trade file sum Realized_PnL:", total_realized_pnl)
print("difference:", total_realized_pnl - final_cum_pnl)
print("does difference == total commission charged (gross PnL - fees = net)?", total_commission_legs)
print("gross PnL would be net + commission:", total_realized_pnl + total_commission_legs, " vs equity says:", final_cum_pnl)

print("\n" + "="*80, "\nDAILY PNL CROSS-CHECK: trade-file daily sum vs equity file Daily_PnL\n", "="*80)
daily_from_trades = exits_new.groupby("Trade_Date")["Realized_PnL"].sum().rename("pnl_from_trades")
eq_idx = eq.set_index("Date")["Daily_PnL"].rename("pnl_from_equity_file")
comp = pd.concat([daily_from_trades, eq_idx], axis=1)
comp_nonzero_trade_days = comp.dropna(subset=["pnl_from_trades"])
mismatch = comp_nonzero_trade_days[(comp_nonzero_trade_days.pnl_from_trades - comp_nonzero_trade_days.pnl_from_equity_file).abs() > 0.01]
print("trade-dates with a closed trade:", len(comp_nonzero_trade_days))
print("of those, mismatching equity-file Daily_PnL:", len(mismatch))
print(mismatch.head(15))
print("\nmax abs mismatch:", (comp_nonzero_trade_days.pnl_from_trades - comp_nonzero_trade_days.pnl_from_equity_file).abs().max())

print("\n" + "="*80, "\nEQUITY FILE DATE COVERAGE / GAPS\n", "="*80)
print("equity file date range:", eq.Date.min(), "->", eq.Date.max())
print("n rows:", len(eq), " n unique dates:", eq.Date.nunique())
all_bdays = pd.bdate_range(eq.Date.min(), eq.Date.max())
missing_bdays = all_bdays.difference(eq.Date)
print("business days in range:", len(all_bdays), " missing business days from equity file:", len(missing_bdays))
print(missing_bdays[:10])

print("\n" + "="*80, "\nDAYS WHERE TRADE FILE SHOWS A CLOSED TRADE BUT EQUITY FILE HAS NO ROW\n", "="*80)
trade_dates_with_exit = set(exits_new.Trade_Date)
eq_dates = set(eq.Date)
missing_from_eq = sorted(trade_dates_with_exit - eq_dates)
print("count:", len(missing_from_eq))
print(missing_from_eq[:10])

print("\n" + "="*80, "\nCUMULATIVE_PNL INTERNAL CONSISTENCY (cumsum of Daily_PnL == Cumulative_PnL?)\n", "="*80)
recomputed = eq.Daily_PnL.cumsum()
bad = (recomputed - eq.Cumulative_PnL).abs() > 0.01
print("rows where cumsum doesn't match stated Cumulative_PnL:", bad.sum())

print("\n" + "="*80, "\nEQUITY_NAV CONSISTENCY (50000 + Cumulative_PnL == Equity_NAV?)\n", "="*80)
bad2 = ((50000 + eq.Cumulative_PnL) - eq.Equity_NAV).abs() > 0.01
print("mismatches:", bad2.sum())

print("\n" + "="*80, "\nDAILY RETURN SERIES FROM EQUITY_NAV (what we'd actually use)\n", "="*80)
eq = eq.sort_values("Date").reset_index(drop=True)
eq["prev_nav"] = eq.Equity_NAV.shift(1)
eq.loc[0, "prev_nav"] = 50000.0
eq["ret"] = eq.Daily_PnL / eq.prev_nav
print(eq["ret"].describe())
print("annualized vol (approx, 252d):", eq["ret"].std()*np.sqrt(252))
cum_ret = (1+eq["ret"]).prod() - 1
n_years = (eq.Date.iloc[-1] - eq.Date.iloc[0]).days/365.25
print(f"total compounded return: {cum_ret:.2%} over {n_years:.2f} years -> CAGR {(1+cum_ret)**(1/n_years)-1:.2%}")
sharpe = eq["ret"].mean()/eq["ret"].std()*np.sqrt(252)
print("naive daily Sharpe (rf=0):", sharpe)
