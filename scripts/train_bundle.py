"""Train a scoring bundle on a whole trade log and save it for the API.

    python scripts/train_bundle.py                                  # synthetic log -> model.joblib
    python scripts/train_bundle.py --trades my_log.csv --out model.joblib
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metalabel.data import load_trade_log, make_synthetic_log, pair_trades  # noqa: E402
from metalabel.features import build_feature_table  # noqa: E402
from metalabel.scoring import save_bundle, train_bundle  # noqa: E402
from metalabel.walkforward import MODELS  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trades", help="fill-log CSV (default: generate synthetic)")
    ap.add_argument("--model", default="hgb", choices=sorted(MODELS))
    ap.add_argument("--keep", type=float, default=0.75, help="fraction of trades to keep (fixed in advance)")
    ap.add_argument("--out", default="model.joblib")
    args = ap.parse_args(argv)

    trades = load_trade_log(args.trades) if args.trades else pair_trades(make_synthetic_log())
    bundle = train_bundle(build_feature_table(trades, mode="t1"), model=args.model, keep_frac=args.keep)
    save_bundle(bundle, args.out)
    print(f"trained {args.model} on {bundle['n_train']} trades, cutoff {bundle['cutoff']:.3f}, wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
