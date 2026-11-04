import os
import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from metalabel.data import make_synthetic_log, pair_trades  # noqa: E402
from metalabel.features import build_feature_table  # noqa: E402
from metalabel.scoring import save_bundle, train_bundle  # noqa: E402


def test_api_flow(tmp_path):
    tr = pair_trades(make_synthetic_log(n_trades=300, seed=5))
    path = tmp_path / "m.joblib"
    save_bundle(train_bundle(build_feature_table(tr)), path)
    os.environ.update(MODEL_BUNDLE=str(path), API_KEYS="k1")
    from api.app import app
    c = TestClient(app)
    assert c.post("/score", json={}, headers={"X-API-Key": "bad"}).status_code == 401
    hist = [{"entry_dt": str(r.entry_dt), "exit_dt": str(r.exit_dt), "exit_price": r.exit_price, "pnl": r.pnl}
            for r in tr.itertuples()]
    assert c.post("/ingest", json=hist, headers={"X-API-Key": "k1"}).status_code == 200
    entry = str(tr["exit_dt"].max() + pd.Timedelta(days=2))
    r = c.post("/score", json={"entry_dt": entry, "entry_price": 1500, "take_profit": 1515},
               headers={"X-API-Key": "k1"})
    assert r.status_code == 200, r.text
    assert set(r.json()) == {"win_probability", "take_trade", "cutoff"}
