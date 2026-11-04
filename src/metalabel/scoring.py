"""Train a production bundle and score a single new trade."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from .features import FEATURES, entry_features, prior_history
from .walkforward import make_model


def train_bundle(table: pd.DataFrame, model: str = "hgb", keep_frac: float = 0.75) -> dict:
    """Fit on all rows and fix the take/skip cutoff from the same training probabilities."""
    clf = make_model(model).fit(table[FEATURES].to_numpy(), table["win"].to_numpy())
    p = clf.predict_proba(table[FEATURES].to_numpy())[:, 1]
    return {"pipeline": clf, "features": FEATURES, "cutoff": float(np.quantile(p, 1 - keep_frac)),
            "keep_frac": keep_frac, "n_train": int(len(table))}


def save_bundle(bundle: dict, path) -> None:
    joblib.dump(bundle, path)


def load_bundle(path) -> dict:
    return joblib.load(path)


def score_trade(bundle: dict, entry_dt, entry_price: float, take_profit: float,
                history: pd.DataFrame) -> dict:
    """Score one trade at entry using only prior-day closed trades (T-1 strict)."""
    prior = prior_history(history, entry_dt, mode="t1")
    f = entry_features(entry_dt, entry_price, take_profit, prior)
    missing = [k for k, v in f.items() if pd.isna(v)]
    if missing:
        raise ValueError(f"not enough prior-day history to compute: {missing}")
    x = np.array([[f[k] for k in bundle["features"]]])
    p = float(bundle["pipeline"].predict_proba(x)[0, 1])
    return {"win_probability": p, "take_trade": p >= bundle["cutoff"], "cutoff": bundle["cutoff"]}
