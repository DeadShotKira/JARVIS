"""Shared document and retrieval data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LoadedDocument:
    """Text extracted from one source file."""

    source_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DocumentChunk:
    """A searchable slice of a source document."""

    id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    """One vector search hit returned by a vector store."""

    chunk: DocumentChunk
    score: float
