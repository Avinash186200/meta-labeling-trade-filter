import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metalabel.data import make_synthetic_log, merge_incremental, pair_trades  # noqa: E402
from metalabel.features import FEATURES, build_feature_table, prior_history  # noqa: E402
from metalabel.metrics import compare  # noqa: E402
from metalabel.scoring import score_trade, train_bundle  # noqa: E402
from metalabel.walkforward import walk_forward  # noqa: E402


@pytest.fixture(scope="module")
def trades():
    return pair_trades(make_synthetic_log(n_trades=600, seed=1))


def test_pairing_one_row_per_position(trades):
    assert trades["position_id"].is_unique
    assert len(trades) == 600
    assert (trades["exit_dt"] >= trades["entry_dt"]).all()


def test_open_position_dropped():
    raw = make_synthetic_log(n_trades=5, seed=2)
    raw = raw.iloc[:-1]  # last position has no exit row
    assert len(pair_trades(raw)) == 4


def test_merge_is_idempotent(trades):
    assert len(merge_incremental(trades, trades)) == len(trades)


def test_t1_history_excludes_same_day(trades):
    t = trades.iloc[300]
    prior = prior_history(trades, t.entry_dt, mode="t1")
    assert (prior["exit_dt"] < t.entry_dt.normalize()).all()


def test_prev_duration_uses_timestamps(trades):
    table = build_feature_table(trades, mode="sequential")
    assert (table["prev_duration_min"] > 0).all()
    assert (table["prev_duration_min"] % (24 * 60) != 0).any()  # not whole days


def test_features_do_not_see_own_outcome(trades):
    table = build_feature_table(trades)
    for col in FEATURES:
        assert abs(np.corrcoef(table[col], table["pnl"])[0, 1]) < 0.95, col


def test_cutoff_comes_from_training_only(trades):
    table = build_feature_table(trades)
    oos = walk_forward(table, keep_frac=0.75)
    # one cutoff per test year, fixed before that year is seen
    assert oos.groupby("test_year")["cutoff"].nunique().eq(1).all()


def test_training_labels_closed_before_test_year(trades):
    """No training trade may still be open when its test year begins."""
    import metalabel.walkforward as wf

    table = build_feature_table(trades).copy()
    # force one late-December trade to close in January
    dec = table.index[(table["entry_dt"].dt.month == 12) & (table["entry_dt"].dt.year == 2018)][-1]
    table.loc[dec, "exit_dt"] = pd.Timestamp("2019-01-03 10:00")
    seen = []
    real = wf.make_model

    class Spy:
        def __init__(self, name):
            self.m = real(name)

        def fit(self, X, y):
            seen.append(len(X))
            self.m.fit(X, y)
            return self

        def predict_proba(self, X):
            return self.m.predict_proba(X)

    wf.make_model = Spy
    try:
        walk_forward(table, keep_frac=0.75)
    finally:
        wf.make_model = real
    start = pd.Timestamp("2019-01-01")
    closed_before = int(((table["entry_dt"] < start) & (table["exit_dt"] < start)).sum())
    entered_before = int((table["entry_dt"] < start).sum())
    years = sorted(table["entry_dt"].dt.year.unique())[2:]  # test years after warm-up
    assert seen[years.index(2019)] == closed_before < entered_before


def test_no_edge_on_pure_noise():
    """With signal_strength=0 a leak-free pipeline must not find an edge."""
    tr = pair_trades(make_synthetic_log(n_trades=900, seed=3, signal_strength=0.0))
    oos = walk_forward(build_feature_table(tr), model="logreg", keep_frac=0.5)
    res = compare(oos)
    lift = res.loc["filtered", "win_rate"] - res.loc["baseline", "win_rate"]
    assert abs(lift) < 0.08


def test_score_trade_roundtrip(trades):
    bundle = train_bundle(build_feature_table(trades))
    entry = trades["exit_dt"].max() + pd.Timedelta(days=2)
    r = score_trade(bundle, entry, 1500.0, 1515.0, trades)
    assert 0 <= r["win_probability"] <= 1 and isinstance(r["take_trade"], bool)


def test_train_bundle_script(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import train_bundle

    out = tmp_path / "model.joblib"
    assert train_bundle.main(["--out", str(out)]) == 0
    assert out.exists()
