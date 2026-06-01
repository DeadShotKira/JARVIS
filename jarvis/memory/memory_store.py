"""SQLite-backed persistent memory storage."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jarvis.database.sqlite_database import connect_sqlite
from jarvis.memory.memory_types import MemoryCandidate, MemoryType


@dataclass(frozen=True)
class MemoryRecord:
    """A durable memory stored in SQLite."""

    id: int
    memory_type: MemoryType
    content: str
    source_text: str
    importance: int
    created_at: str
    updated_at: str
    access_count: int
    metadata: dict[str, Any]


class MemoryStore:
    """Stores structured memories in SQLite.

    The schema also includes a separate memory_embeddings table so Phase 3 can
    add semantic search without redesigning the database.
    """

    def __init__(self, database_path: Path | str):
        self.connection = connect_sqlite(database_path)
        self.initialize()

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                source_text TEXT NOT NULL DEFAULT '',
                importance INTEGER NOT NULL DEFAULT 3,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_accessed_at TEXT,
                access_count INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                embedding_status TEXT NOT NULL DEFAULT 'pending'
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_type_content
                ON memories(memory_type, content);

            CREATE INDEX IF NOT EXISTS idx_memories_type
                ON memories(memory_type);

            CREATE TABLE IF NOT EXISTS memory_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_vector_json TEXT NOT NULL,
                embedding_dimensions INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                UNIQUE(memory_id, embedding_model)
            );
            """
        )
        self.connection.commit()

    def add_candidate(self, candidate: MemoryCandidate) -> MemoryRecord:
        return self.add_memory(
            memory_type=candidate.memory_type,
            content=candidate.content,
            source_text=candidate.source_text,
            importance=candidate.importance,
        )

    def add_memory(
        self,
        memory_type: MemoryType,
        content: str,
        source_text: str = "",
        importance: int = 3,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        now = _utc_now()
        metadata_json = json.dumps(metadata or {}, sort_keys=True)

        try:
            cursor = self.connection.execute(
                """
                INSERT INTO memories (
                    memory_type,
                    content,
                    source_text,
                    importance,
                    created_at,
                    updated_at,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_type.value,
                    content,
                    source_text,
                    importance,
                    now,
                    now,
                    metadata_json,
                ),
            )
            self.connection.commit()
            return self.get_memory(cursor.lastrowid)
        except sqlite3.IntegrityError:
            self.connection.execute(
                """
                UPDATE memories
                   SET updated_at = ?,
                       source_text = ?,
                       importance = MAX(importance, ?)
                 WHERE memory_type = ?
                   AND content = ?
                """,
                (now, source_text, importance, memory_type.value, content),
            )
            self.connection.commit()
            return self.get_by_type_and_content(memory_type, content)

    def get_memory(self, memory_id: int) -> MemoryRecord:
        row = self.connection.execute(
            "SELECT * FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Memory not found: {memory_id}")
        return _row_to_record(row)

    def get_by_type_and_content(self, memory_type: MemoryType, content: str) -> MemoryRecord:
        row = self.connection.execute(
            "SELECT * FROM memories WHERE memory_type = ? AND content = ?",
            (memory_type.value, content),
        ).fetchone()
        if row is None:
            raise KeyError(f"Memory not found: {memory_type.value} {content!r}")
        return _row_to_record(row)

    def list_memories(self) -> list[MemoryRecord]:
        rows = self.connection.execute(
            "SELECT * FROM memories ORDER BY importance DESC, updated_at DESC, id DESC"
        ).fetchall()
        return [_row_to_record(row) for row in rows]

    def list_by_type(self, memory_type: MemoryType) -> list[MemoryRecord]:
        rows = self.connection.execute(
            """
            SELECT * FROM memories
             WHERE memory_type = ?
             ORDER BY importance DESC, updated_at DESC, id DESC
            """,
            (memory_type.value,),
        ).fetchall()
        return [_row_to_record(row) for row in rows]

    def touch_memories(self, memory_ids: list[int]) -> None:
        if not memory_ids:
            return
        now = _utc_now()
        self.connection.executemany(
            """
            UPDATE memories
               SET last_accessed_at = ?,
                   access_count = access_count + 1
             WHERE id = ?
            """,
            [(now, memory_id) for memory_id in memory_ids],
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=int(row["id"]),
        memory_type=MemoryType(row["memory_type"]),
        content=str(row["content"]),
        source_text=str(row["source_text"]),
        importance=int(row["importance"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        access_count=int(row["access_count"]),
        metadata=json.loads(row["metadata_json"]),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
