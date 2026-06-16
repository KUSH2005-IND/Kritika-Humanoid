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

    def get_session_duration(self, name: str, date: str = None) -> dict:
        """
        Returns entry/exit pairs for the most recent session for a given person.
        If no exit is recorded (person still present), last_seen = last entry.
        """
        date_filter = f"AND date(ts, 'unixepoch') = '{date}'" if date else ""
        query = f"""
        SELECT event, ts FROM presence_log
        WHERE name = ? {date_filter}
        ORDER BY ts ASC
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(query, (name,)).fetchall()
            if not rows:
                return {}
            # Find last entry/exit pair
            last_entry = None
            last_exit = None
            for event, ts in rows:
                if event == 'entry':
                    last_entry = ts
                    last_exit = None   # reset — new session started
                elif event == 'exit' and last_entry is not None:
                    last_exit = ts
            if last_entry is None:
                return {}
            end_ts = last_exit if last_exit else last_entry
            return {
                'arrived': last_entry,
                'last_seen': end_ts,
                'minutes': (end_ts - last_entry) / 60.0,
                'still_present': last_exit is None
            }
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
