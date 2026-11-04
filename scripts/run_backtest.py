"""End-to-end run: trade log -> features -> walk-forward -> results.

    python scripts/run_backtest.py                      # synthetic data
    python scripts/run_backtest.py --trades my_log.csv  # your own export
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metalabel.data import load_trade_log, make_synthetic_log, pair_trades  # noqa: E402
from metalabel.features import build_feature_table  # noqa: E402
from metalabel.metrics import compare  # noqa: E402
from metalabel.walkforward import MODELS, walk_forward  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trades", help="fill-log CSV (default: generate synthetic)")
    ap.add_argument("--mode", default="t1", choices=["t1", "sequential"])
    ap.add_argument("--keep", type=float, default=0.75, help="fraction of trades to keep (fixed in advance)")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args(argv)

    trades = load_trade_log(args.trades) if args.trades else pair_trades(make_synthetic_log())
    table = build_feature_table(trades, mode=args.mode)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = {"input": args.trades or "synthetic (seed 7)", "mode": args.mode,
               "keep_frac": args.keep, "n_trades": len(trades), "n_model_rows": len(table), "models": {}}
    for name in MODELS:
        oos = walk_forward(table, model=name, keep_frac=args.keep)
        res = compare(oos)
        res.to_csv(out / f"oos_{name}.csv")
        summary["models"][name] = res.round(4).to_dict(orient="index")
        print(f"\n== {name} ==\n{res.round(3).to_string()}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
