"""Memory category definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MemoryType(str, Enum):
    """Supported long-term memory categories."""

    PREFERENCE = "PREFERENCE"
    INTEREST = "INTEREST"
    PROJECT = "PROJECT"
    TASK = "TASK"
    FACT = "FACT"


@dataclass(frozen=True)
class MemoryCandidate:
    """A memory extracted from a user message before database insertion."""

    memory_type: MemoryType
    content: str
    source_text: str
    importance: int = 3
