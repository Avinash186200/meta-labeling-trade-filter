import pandas as pd
import numpy as np

path = "<local-path>"
df = pd.read_csv(path, parse_dates=["Trade_Date"])

print("="*80)
print("BASIC SHAPE / COLUMNS")
print("="*80)
print(df.shape)
print(df.dtypes)

print("\n" + "="*80)
print("UNIQUE KEY FIELDS")
print("="*80)
for c in ["Account_ID","Strategy_ID","Symbol","Asset_Class","Exchange","Currency","Side","Order_Type","Entry_Exit","Timezone","Leverage"]:
    print(c, "->", df[c].unique())

print("\n" + "="*80)
print("DATE RANGE")
print("="*80)
print("min:", df.Trade_Date.min(), "max:", df.Trade_Date.max())
print("distinct trade dates:", df.Trade_Date.nunique())

print("\n" + "="*80)
print("MISSING VALUE COUNTS")
print("="*80)
print(df.isna().sum())

print("\n" + "="*80)
print("ENTRY/EXIT COUNTS")
print("="*80)
print(df.Entry_Exit.value_counts())

print("\n" + "="*80)
print("POSITION ID INTEGRITY (each position should have 1 entry + 1 exit)")
print("="*80)
pos_counts = df.groupby("Position_ID")["Entry_Exit"].apply(list)
bad = pos_counts[pos_counts.apply(lambda x: sorted(x) != ["Entry","Exit"])]
print("positions not exactly [Entry,Exit]:", len(bad), "/", len(pos_counts))
if len(bad):
    print(bad.head(10))

print("\n" + "="*80)
print("REALIZED PNL STATS (exits only)")
print("="*80)
exits = df[df.Entry_Exit=="Exit"].copy()
print(exits.Realized_PnL.describe())
print("sum PnL:", exits.Realized_PnL.sum())
print("count exits with null PnL:", exits.Realized_PnL.isna().sum())

print("\n" + "="*80)
print("COMMISSION / FEES / SETTLEMENT_DATE / STOP_LOSS / TAKE_PROFIT / MARGIN_USED null %")
print("="*80)
for c in ["Commission","Fees","Settlement_Date","Stop_Loss","Take_Profit","Margin_Used","Avg_Entry_Price","Open_Position_Qty_After"]:
    print(c, "null%:", round(100*df[c].isna().mean(),1))

print("\n" + "="*80)
print("NOTES VALUE COUNTS (exit reasons)")
print("="*80)
print(df.Notes.value_counts(dropna=False))

print("\n" + "="*80)
print("TRADE DURATION CHECK: gaps between consecutive trade dates")
print("="*80)
dates = pd.Series(sorted(df.Trade_Date.unique()))
gaps = dates.diff().dt.days
print(gaps.describe())
print("max gap days:", gaps.max(), "at date:", dates[gaps.idxmax()])

print("\n" + "="*80)
print("DAILY PNL AGGREGATION (sum Realized_PnL by exit Trade_Date)")
print("="*80)
daily_pnl = exits.groupby("Trade_Date")["Realized_PnL"].sum().sort_index()
print("days with at least one closed trade:", len(daily_pnl))
print(daily_pnl.describe())

print("\n" + "="*80)
print("NOTIONAL / LEVERAGE / MARGIN — for converting PnL to a RETURN")
print("="*80)
print(df.Notional_Value.describe())
print("Quantity unique:", df.Quantity.unique())
print("Margin_Used non-null sample:\n", df.Margin_Used.dropna().head())

print("\n" + "="*80)
print("YEARLY TRADE COUNT / PNL")
print("="*80)
exits["year"] = exits.Trade_Date.dt.year
print(exits.groupby("year")["Realized_PnL"].agg(["count","sum","mean"]))
