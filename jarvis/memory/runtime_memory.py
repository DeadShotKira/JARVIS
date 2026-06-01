"""Runtime-only conversation memory."""

from __future__ import annotations

from dataclasses import dataclass, field

from jarvis.brain.ollama_client import ChatMessage


@dataclass
class RuntimeMemory:
    """Stores recent messages in RAM only.

    This deliberately stores recent chat turns only. Long-term memory is handled
    by MemoryManager and SQLite.
    """

    max_messages: int = 10
    messages: list[ChatMessage] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append(ChatMessage(role=role, content=content))
        self._trim()

    def clear(self) -> None:
        self.messages.clear()

    def _trim(self) -> None:
        overflow = len(self.messages) - self.max_messages
        if overflow > 0:
            del self.messages[:overflow]
