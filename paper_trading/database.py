"""SQLite persistence layer for paper trading state."""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path("paper_trading.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS agents (
            name TEXT PRIMARY KEY,
            config TEXT NOT NULL,
            starting_balance REAL NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cash (
            agent_name TEXT PRIMARY KEY,
            amount REAL NOT NULL,
            FOREIGN KEY (agent_name) REFERENCES agents(name)
        );

        CREATE TABLE IF NOT EXISTS positions (
            agent_name TEXT NOT NULL,
            ticker TEXT NOT NULL,
            shares REAL NOT NULL,
            avg_cost REAL NOT NULL,
            PRIMARY KEY (agent_name, ticker),
            FOREIGN KEY (agent_name) REFERENCES agents(name)
        );

        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            shares REAL NOT NULL,
            price REAL NOT NULL,
            total_value REAL NOT NULL,
            conviction_weight REAL,
            notes TEXT,
            FOREIGN KEY (agent_name) REFERENCES agents(name)
        );

        CREATE TABLE IF NOT EXISTS daily_snapshots (
            agent_name TEXT NOT NULL,
            date TEXT NOT NULL,
            cash REAL NOT NULL,
            positions_value REAL NOT NULL,
            portfolio_value REAL NOT NULL,
            daily_return_pct REAL,
            cumulative_return_pct REAL NOT NULL,
            PRIMARY KEY (agent_name, date),
            FOREIGN KEY (agent_name) REFERENCES agents(name)
        );
        """)


def register_agent(name, config_dict, starting_balance):
    """Insert or replace an agent record and initialise its cash."""
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO agents (name, config, starting_balance, created_at) VALUES (?, ?, ?, ?)",
            (name, json.dumps(config_dict), starting_balance, datetime.now().isoformat()),
        )
        conn.execute(
            "INSERT OR IGNORE INTO cash (agent_name, amount) VALUES (?, ?)",
            (name, starting_balance),
        )


def get_agent(name):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM agents WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None


def get_all_agents():
    with get_connection() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM agents ORDER BY name").fetchall()]


def get_cash(agent_name):
    with get_connection() as conn:
        row = conn.execute("SELECT amount FROM cash WHERE agent_name = ?", (agent_name,)).fetchone()
        return float(row["amount"]) if row else 0.0


def set_cash(agent_name, amount, conn=None):
    sql = "INSERT OR REPLACE INTO cash (agent_name, amount) VALUES (?, ?)"
    if conn:
        conn.execute(sql, (agent_name, amount))
    else:
        with get_connection() as c:
            c.execute(sql, (agent_name, amount))


def get_positions(agent_name):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT ticker, shares, avg_cost FROM positions WHERE agent_name = ?", (agent_name,)
        ).fetchall()
        return {r["ticker"]: {"shares": float(r["shares"]), "avg_cost": float(r["avg_cost"])} for r in rows}


def upsert_position(agent_name, ticker, shares, avg_cost, conn=None):
    if shares <= 0:
        sql_del = "DELETE FROM positions WHERE agent_name = ? AND ticker = ?"
        if conn:
            conn.execute(sql_del, (agent_name, ticker))
        else:
            with get_connection() as c:
                c.execute(sql_del, (agent_name, ticker))
    else:
        sql_up = "INSERT OR REPLACE INTO positions (agent_name, ticker, shares, avg_cost) VALUES (?, ?, ?, ?)"
        if conn:
            conn.execute(sql_up, (agent_name, ticker, shares, avg_cost))
        else:
            with get_connection() as c:
                c.execute(sql_up, (agent_name, ticker, shares, avg_cost))


def record_trade(agent_name, date, ticker, action, shares, price, conviction_weight=None, notes=None):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO trades (agent_name, date, ticker, action, shares, price, total_value, conviction_weight, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (agent_name, date, ticker, action, shares, price, shares * price, conviction_weight, notes),
        )


def record_snapshot(agent_name, date, cash, positions_value, starting_balance):
    portfolio_value = cash + positions_value
    cumulative_return = (portfolio_value - starting_balance) / starting_balance * 100.0
    with get_connection() as conn:
        prev = conn.execute(
            "SELECT portfolio_value FROM daily_snapshots WHERE agent_name = ? ORDER BY date DESC LIMIT 1",
            (agent_name,),
        ).fetchone()
        daily_return = None
        if prev:
            prev_val = float(prev["portfolio_value"])
            daily_return = (portfolio_value - prev_val) / prev_val * 100.0 if prev_val else None
        conn.execute(
            "INSERT OR REPLACE INTO daily_snapshots (agent_name, date, cash, positions_value, portfolio_value, daily_return_pct, cumulative_return_pct) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (agent_name, date, cash, positions_value, portfolio_value, daily_return, cumulative_return),
        )


def get_latest_snapshot(agent_name):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM daily_snapshots WHERE agent_name = ? ORDER BY date DESC LIMIT 1",
            (agent_name,),
        ).fetchone()
        return dict(row) if row else None


def get_all_snapshots(agent_name):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM daily_snapshots WHERE agent_name = ? ORDER BY date",
            (agent_name,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_trades(agent_name):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE agent_name = ? ORDER BY date, id",
            (agent_name,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_leaderboard():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                a.name,
                a.starting_balance,
                COALESCE(s.portfolio_value, a.starting_balance) AS portfolio_value,
                COALESCE(s.cash, a.starting_balance) AS cash,
                COALESCE(s.positions_value, 0) AS positions_value,
                COALESCE(s.cumulative_return_pct, 0.0) AS cumulative_return_pct,
                s.daily_return_pct,
                s.date AS last_updated
            FROM agents a
            LEFT JOIN daily_snapshots s
                ON a.name = s.agent_name
                AND s.date = (
                    SELECT MAX(date) FROM daily_snapshots WHERE agent_name = a.name
                )
            ORDER BY cumulative_return_pct DESC
        """).fetchall()
        return [dict(r) for r in rows]
