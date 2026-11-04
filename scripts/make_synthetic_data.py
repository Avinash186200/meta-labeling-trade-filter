"""Write a synthetic fill log to data/sample_trades.csv (same columns as a real export)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from metalabel.data import make_synthetic_log  # noqa: E402

if __name__ == "__main__":
    (ROOT / "data").mkdir(exist_ok=True)
    df = make_synthetic_log()
    df.to_csv(ROOT / "data" / "sample_trades.csv", index=False)
    print(f"wrote {len(df)} fill rows to data/sample_trades.csv")
