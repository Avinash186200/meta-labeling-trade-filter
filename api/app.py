"""Minimal scoring service.

    MODEL_BUNDLE=model.joblib API_KEYS=key1,key2 uvicorn api.app:app

POST /ingest  closed trades for the caller's account (kept in memory per key)
POST /score   one new trade -> win probability + take/skip
Keys come from the environment; nothing is stored on disk.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metalabel.scoring import load_bundle, score_trade  # noqa: E402

app = FastAPI(title="Meta-labeling trade filter")
_HISTORY: dict[str, pd.DataFrame] = {}
_BUNDLE = None


def bundle() -> dict:
    """Lazily load the model bundle named by MODEL_BUNDLE."""
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = load_bundle(os.environ["MODEL_BUNDLE"])
    return _BUNDLE


def account(x_api_key: str = Header(...)) -> str:
    """Resolve the caller from the X-API-Key header; 401 if unknown."""
    keys = {k for k in os.environ.get("API_KEYS", "").split(",") if k}
    if x_api_key not in keys:
        raise HTTPException(401, "invalid API key")
    return x_api_key


class ClosedTrade(BaseModel):
    entry_dt: datetime
    exit_dt: datetime
    exit_price: float
    pnl: float


class NewTrade(BaseModel):
    entry_dt: datetime
    entry_price: float
    take_profit: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(trades: list[ClosedTrade], acct: str = Depends(account)):
    new = pd.DataFrame([t.model_dump() for t in trades])
    new[["entry_dt", "exit_dt"]] = new[["entry_dt", "exit_dt"]].apply(pd.to_datetime)
    hist = pd.concat([_HISTORY.get(acct, pd.DataFrame()), new], ignore_index=True)
    _HISTORY[acct] = hist.drop_duplicates().sort_values("exit_dt").reset_index(drop=True)
    return {"stored": len(_HISTORY[acct])}


@app.post("/score")
def score(trade: NewTrade, acct: str = Depends(account)):
    hist = _HISTORY.get(acct)
    if hist is None or hist.empty:
        raise HTTPException(409, "no history ingested for this account")
    try:
        return score_trade(bundle(), trade.entry_dt, trade.entry_price, trade.take_profit, hist)
    except ValueError as e:
        raise HTTPException(422, str(e))
