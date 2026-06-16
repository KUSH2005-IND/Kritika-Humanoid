# presence/manager.py
"""
Presence Manager — tracks who is currently in the scene.
Fires entry/exit events and logs to SQLite.
"""

import time
import sqlite3
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PRESENCE_TIMEOUT_SEC, DB_PATH


class PresenceManager:
    """
    Maintains who is currently in the scene.
    Fires entry/exit events and logs to SQLite.
    """

    def __init__(self, db_path=DB_PATH):
        self.active: dict[str, float] = {}    # {name: last_seen_timestamp}
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS presence_log (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    name    TEXT NOT NULL,
                    event   TEXT NOT NULL,    -- 'entry' or 'exit'
                    ts      REAL NOT NULL
                )
            """)
            # Create indices if they don't exist
            conn.execute("CREATE INDEX IF NOT EXISTS idx_presence_name ON presence_log(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_presence_ts ON presence_log(ts)")

    def update(self, identities: list[str]) -> dict:
        """
        Call every frame with list of currently detected identity names.
        Returns {'entries': [...], 'exits': [...]}
        """
        now = time.time()
        events = {'entries': [], 'exits': []}

        for name in identities:
            if name == 'Unknown' or name is None:
                continue
            if name not in self.active:
                events['entries'].append(name)
                self._log(name, 'entry', now)
            self.active[name] = now

        # Detect exits
        for name in list(self.active.keys()):
            if now - self.active[name] > PRESENCE_TIMEOUT_SEC:
                events['exits'].append(name)
                self._log(name, 'exit', now)
                del self.active[name]

        return events

    def _log(self, name: str, event: str, ts: float):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO presence_log (name, event, ts) VALUES (?, ?, ?)",
                    (name, event, ts)
                )
        except sqlite3.Error as e:
            print(f"[Presence] DB write error: {e}")

    @property
    def present(self) -> list[str]:
        """Currently present known individuals."""
        return sorted(self.active.keys())

    def get_history(self, limit=50) -> list[dict]:
        """Get recent presence history from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT name, event, ts FROM presence_log ORDER BY ts DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [{'name': r[0], 'event': r[1], 'ts': r[2]} for r in rows]
        except sqlite3.Error as e:
            print(f"[Presence] DB read error: {e}")
            return []
