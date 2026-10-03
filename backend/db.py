import os
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = "/var/lib/worklog/worklog.db"
DB_PATH = Path(os.getenv("WORKLOG_DB_PATH", DEFAULT_DB_PATH))


def get_connection(read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        wal_path = Path(f"{DB_PATH}-wal")
        db_uri = f"{DB_PATH.resolve().as_uri()}?mode=ro"

        # With no WAL data, immutable mode lets SQLite read the
        # database from the API container's read-only mount.
        if not wal_path.exists() or wal_path.stat().st_size == 0:
            db_uri += "&immutable=1"

        conn = sqlite3.connect(db_uri, uri=True, timeout=30)
    else:
        conn = sqlite3.connect(DB_PATH, timeout=30)

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn