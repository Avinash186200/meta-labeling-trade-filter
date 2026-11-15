"""
GC Meta-model overlay -- live scoring API.

Two endpoints:
  POST /ingest  -- the caller pushes closed trades (their daily PnL/trade export)
                   to build up the rolling history the model needs.
  POST /score   -- the caller pushes a new trade about to be entered
                   (entry_price, take_profit, entry_dt) and gets back a
                   win-probability + take/skip decision.

Auth: one API key per account, sent as `X-API-Key` header. Create accounts
with `python manage.py create-account <account_id>` (see manage.py).

Run locally for testing:
    uvicorn app:app --reload --port 8000
"""
import sys
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))  # so `import predict_live` works
from predict_live import score_new_trade  # noqa: E402

import storage

storage.init_db()
app = FastAPI(title="GC Meta-model Overlay API", version="1.0")


def authenticate(x_api_key: str) -> str:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    account = storage.get_account_by_key(x_api_key)
    if account is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return account["account_id"]


class Trade(BaseModel):
    entry_dt: datetime
    exit_dt: datetime
    exit_date: datetime
    exit_price: float
    pnl: float


class IngestRequest(BaseModel):
    trades: list[Trade] = Field(..., min_length=1)


class ScoreRequest(BaseModel):
    entry_price: float
    take_profit: float
    entry_dt: datetime


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest, x_api_key: str = Header(None)):
    account_id = authenticate(x_api_key)
    rows = [
        {
            "entry_dt": t.entry_dt.isoformat(),
            "exit_dt": t.exit_dt.isoformat(),
            "exit_date": t.exit_date.date().isoformat(),
            "exit_price": t.exit_price,
            "pnl": t.pnl,
        }
        for t in req.trades
    ]
    storage.add_trades(account_id, rows)
    return {"status": "ok", "trades_added": len(rows)}


@app.post("/score")
def score(req: ScoreRequest, x_api_key: str = Header(None)):
    account_id = authenticate(x_api_key)
    history = storage.get_history(account_id)
    if history.empty:
        raise HTTPException(status_code=422, detail="No trade history on file yet -- call /ingest first")
    try:
        result = score_new_trade(
            entry_price=req.entry_price,
            take_profit=req.take_profit,
            entry_dt=req.entry_dt,
            history_df=history,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return result
