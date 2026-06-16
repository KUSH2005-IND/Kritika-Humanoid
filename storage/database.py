# storage/database.py
"""
SQLite helpers for the presence database.
Provides querying capabilities over the presence_log table.
"""

import sqlite3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

class DatabaseHelper:
    """
    Helper class to run analytical queries on the presence SQLite database.
    """

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

    def get_presence_today(self) -> list[str]:
        """
        Returns a list of unique names of individuals who have been present today.
        """
        query = """
        SELECT DISTINCT name FROM presence_log
        WHERE event = 'entry' AND ts > strftime('%s', 'now', 'start of day');
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(query).fetchall()
            return [r[0] for r in rows]
        except sqlite3.Error as e:
            print(f"[Database] Error querying presence today: {e}")
            return []

    def get_session_duration(self, name: str) -> dict:
        """
        Returns the duration of the last session for a given person.
        (Note: based on the IMPLEMENTATION.md example, this calculates based on 'entry' events,
        but for a more accurate duration, it could be extended to consider 'exit' events too.)
        """
        query = """
        SELECT
            MIN(ts) AS arrived,
            MAX(ts) AS last_seen,
            (MAX(ts) - MIN(ts)) / 60.0 AS minutes
        FROM presence_log
        WHERE name = ? AND event = 'entry';
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(query, (name,)).fetchone()
            if row and row[0] is not None:
                return {
                    'arrived': row[0],
                    'last_seen': row[1],
                    'minutes': row[2]
                }
            return {}
        except sqlite3.Error as e:
            print(f"[Database] Error querying session duration for {name}: {e}")
            return {}

    def get_recent_history(self, limit=50) -> list[dict]:
        """
        Returns the most recent presence events.
        """
        query = "SELECT name, event, ts FROM presence_log ORDER BY ts DESC LIMIT ?"
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(query, (limit,)).fetchall()
            return [{'name': r[0], 'event': r[1], 'ts': r[2]} for r in rows]
        except sqlite3.Error as e:
            print(f"[Database] Error querying history: {e}")
            return []
