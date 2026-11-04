"""Yearly expanding-window walk-forward with a hindsight-free take/skip cutoff.

For each test year Y:
  1. fit scaler + model on all trades entered AND closed before Y (purged)
  2. cutoff = quantile of the model's probabilities on the TRAINING set
     (never on the test year - choosing it from test probabilities would let
     the filter know the future distribution of scores)
  3. score year Y and keep trades with P(win) >= cutoff
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_leaf=20, random_state=42),
    "hgb": lambda: HistGradientBoostingClassifier(
        max_depth=3, max_iter=150, learning_rate=0.05, random_state=42),
}


def make_model(name: str) -> Pipeline:
    """Fresh scaler + classifier pipeline (fitted inside each fold)."""
    return make_pipeline(StandardScaler(), MODELS[name]())


def walk_forward(table: pd.DataFrame, model: str = "hgb", keep_frac: float = 0.75,
                 warmup_years: int = 2, min_train: int = 30) -> pd.DataFrame:
    """Return out-of-sample rows with columns proba, cutoff, keep, test_year."""
    t = table.sort_values("entry_dt").reset_index(drop=True)
    year = t["entry_dt"].dt.year
    years = sorted(year.unique())
    out = []
    for y in years[warmup_years:]:
        # Purge: a training label is only usable if the trade had CLOSED before
        # the test year starts (a late-December entry can exit in January).
        test_start = pd.Timestamp(year=int(y), month=1, day=1)
        tr = t[(year < y) & (t["exit_dt"] < test_start)]
        te = t[year == y]
        if len(tr) < min_train or te.empty:
            continue
        clf = make_model(model).fit(tr[FEATURES].to_numpy(), tr["win"].to_numpy())
        cutoff = float(np.quantile(clf.predict_proba(tr[FEATURES].to_numpy())[:, 1], 1 - keep_frac))
        rec = te.copy()
        rec["proba"] = clf.predict_proba(te[FEATURES].to_numpy())[:, 1]
        rec["cutoff"] = cutoff
        rec["keep"] = rec["proba"] >= cutoff
        rec["test_year"] = y
        out.append(rec)
    if not out:
        raise ValueError("not enough years for a walk-forward")
    return pd.concat(out, ignore_index=True)
