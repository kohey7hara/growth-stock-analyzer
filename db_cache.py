"""
db_cache.py - SQLiteキャッシュ管理モジュール

テーブル:
  daily_prices: ticker, date, open, high, low, close, volume
"""

import sqlite3
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta

DB_PATH = Path(__file__).parent / "data" / "stock_cache.db"

logger = logging.getLogger(__name__)


def get_connection():
    """SQLite接続を取得"""
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """テーブル作成"""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_prices (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticker_date
        ON daily_prices(ticker, date)
    """)
    conn.commit()
    conn.close()


def get_latest_date(ticker):
    """指定ティッカーの最新日付を返す。データなければNone"""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT MAX(date) FROM daily_prices WHERE ticker = ?", (ticker,)
    )
    result = cursor.fetchone()[0]
    conn.close()
    return result


def upsert_daily_prices(ticker, df):
    """日足データをUPSERT (INSERT OR REPLACE)"""
    if df.empty:
        return

    conn = get_connection()
    rows = []
    for idx, row in df.iterrows():
        # yfinanceのインデックスはTimestamp (tz-aware可能)
        if hasattr(idx, 'strftime'):
            date_str = idx.strftime("%Y-%m-%d")
        else:
            date_str = str(idx)[:10]

        rows.append((
            ticker,
            date_str,
            float(row.get("Open", 0)),
            float(row.get("High", 0)),
            float(row.get("Low", 0)),
            float(row.get("Close", 0)),
            int(row.get("Volume", 0)),
        ))

    conn.executemany("""
        INSERT OR REPLACE INTO daily_prices
        (ticker, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()
    logger.debug(f"  SQLite: {ticker} に {len(rows)}行保存")


def load_daily_prices(ticker, days=400):
    """SQLiteからN日分の日足データを取得"""
    conn = get_connection()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM daily_prices "
        "WHERE ticker = ? AND date >= ? ORDER BY date",
        conn,
        params=(ticker, cutoff),
    )
    conn.close()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df.columns = ["Open", "High", "Low", "Close", "Volume"]

    return df


def load_full_history(ticker):
    """SQLiteから全日足データを取得（株価推移計算用）"""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM daily_prices "
        "WHERE ticker = ? ORDER BY date",
        conn,
        params=(ticker,),
    )
    conn.close()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df.columns = ["Open", "High", "Low", "Close", "Volume"]

    return df


def get_row_count(ticker):
    """指定ティッカーの行数を返す"""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT COUNT(*) FROM daily_prices WHERE ticker = ?", (ticker,)
    )
    result = cursor.fetchone()[0]
    conn.close()
    return result
