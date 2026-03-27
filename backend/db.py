import os
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = "/var/lib/worklog/worklog.db"
DB_PATH = Path(os.getenv("WORKLOG_DB_PATH", DEFAULT_DB_PATH))


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn