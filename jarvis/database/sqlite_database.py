"""SQLite connection helper."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def connect_sqlite(database_path: Path | str) -> sqlite3.Connection:
    """Open a SQLite database and enable pragmatic local defaults."""
    if str(database_path) != ":memory:":
        Path(database_path).expanduser().parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection
