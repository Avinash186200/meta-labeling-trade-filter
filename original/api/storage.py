"""SQLite storage for per-account trade history + API keys. One file, no
external DB server needed -- fine for this scale (a handful of accounts,
a few thousand trades each).
"""
import sqlite3
import secrets
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent / "gc_api.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            account_id TEXT PRIMARY KEY,
            api_key TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id TEXT NOT NULL,
            entry_dt TEXT NOT NULL,
            exit_dt TEXT NOT NULL,
            exit_date TEXT NOT NULL,
            exit_price REAL NOT NULL,
            pnl REAL NOT NULL,
            FOREIGN KEY(account_id) REFERENCES accounts(account_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_account ON trades(account_id, exit_date)")
    conn.commit()
    conn.close()


def create_account(account_id: str) -> str:
    api_key = secrets.token_urlsafe(32)
    conn = get_conn()
    conn.execute(
        "INSERT INTO accounts (account_id, api_key, created_at) VALUES (?, ?, ?)",
        (account_id, api_key, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return api_key


def get_account_by_key(api_key: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM accounts WHERE api_key = ?", (api_key,)).fetchone()
    conn.close()
    return row


def add_trades(account_id: str, trades: list[dict]):
    conn = get_conn()
    conn.executemany(
        """INSERT INTO trades (account_id, entry_dt, exit_dt, exit_date, exit_price, pnl)
           VALUES (:account_id, :entry_dt, :exit_dt, :exit_date, :exit_price, :pnl)""",
        [{**t, "account_id": account_id} for t in trades],
    )
    conn.commit()
    conn.close()


def get_history(account_id: str):
    import pandas as pd
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT entry_dt, exit_dt, exit_date, exit_price, pnl FROM trades "
        "WHERE account_id = ? ORDER BY exit_dt ASC",
        conn, params=(account_id,),
        parse_dates=["entry_dt", "exit_dt", "exit_date"],
    )
    conn.close()
    return df
