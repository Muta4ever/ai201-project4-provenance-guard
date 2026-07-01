"""SQLite persistence for Provenance Guard: content store + structured audit log.

Two tables:
  - content:   the latest state of each submission (status, scores).
  - audit_log: an append-only, structured record of every event
               (classifications and appeals), one JSON-friendly row each.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).with_name("provenance.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS content (
                content_id      TEXT PRIMARY KEY,
                creator_id      TEXT NOT NULL,
                text            TEXT NOT NULL,
                attribution     TEXT NOT NULL,
                confidence      REAL NOT NULL,
                llm_score       REAL,
                structural_score REAL,
                lexical_score   REAL,
                status          TEXT NOT NULL,
                created_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id       TEXT NOT NULL,
                creator_id       TEXT,
                timestamp        TEXT NOT NULL,
                event_type       TEXT NOT NULL,      -- 'classified' | 'appeal'
                attribution      TEXT,
                confidence       REAL,
                llm_score        REAL,
                structural_score REAL,
                lexical_score    REAL,
                status           TEXT,
                appeal_reasoning TEXT,
                detail_json      TEXT                -- full signal detail as JSON
            );
            """
        )


def save_content(record: dict) -> None:
    """Insert or replace the current state of a piece of content."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO content
                (content_id, creator_id, text, attribution, confidence,
                 llm_score, structural_score, lexical_score, status, created_at)
            VALUES (:content_id, :creator_id, :text, :attribution, :confidence,
                    :llm_score, :structural_score, :lexical_score, :status, :created_at)
            """,
            record,
        )


def get_content(content_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM content WHERE content_id = ?", (content_id,)
        ).fetchone()
    return dict(row) if row else None


def update_status(content_id: str, status: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE content SET status = ? WHERE content_id = ?", (status, content_id)
        )


def add_audit_entry(entry: dict) -> None:
    """Append a structured entry to the audit log."""
    payload = {
        "content_id": entry.get("content_id"),
        "creator_id": entry.get("creator_id"),
        "timestamp": entry.get("timestamp"),
        "event_type": entry.get("event_type"),
        "attribution": entry.get("attribution"),
        "confidence": entry.get("confidence"),
        "llm_score": entry.get("llm_score"),
        "structural_score": entry.get("structural_score"),
        "lexical_score": entry.get("lexical_score"),
        "status": entry.get("status"),
        "appeal_reasoning": entry.get("appeal_reasoning"),
        "detail_json": json.dumps(entry.get("detail")) if entry.get("detail") else None,
    }
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO audit_log
                (content_id, creator_id, timestamp, event_type, attribution, confidence,
                 llm_score, structural_score, lexical_score, status, appeal_reasoning,
                 detail_json)
            VALUES (:content_id, :creator_id, :timestamp, :event_type, :attribution,
                    :confidence, :llm_score, :structural_score, :lexical_score, :status,
                    :appeal_reasoning, :detail_json)
            """,
            payload,
        )


def get_log(limit: int = 50) -> list[dict]:
    """Return the most recent audit entries (newest first), with detail parsed back."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    entries = []
    for row in rows:
        entry = dict(row)
        if entry.get("detail_json"):
            try:
                entry["detail"] = json.loads(entry["detail_json"])
            except json.JSONDecodeError:
                entry["detail"] = None
        entry.pop("detail_json", None)
        entries.append(entry)
    return entries
