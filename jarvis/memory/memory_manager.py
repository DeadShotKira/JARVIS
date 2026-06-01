"""Persistent memory orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.memory.memory_retriever import MemoryRetriever
from jarvis.memory.memory_store import MemoryRecord, MemoryStore
from jarvis.memory.memory_utils import extract_memory_candidates


@dataclass
class MemoryManager:
    """Coordinates extraction, storage, retrieval, and prompt formatting."""

    store: MemoryStore
    retriever: MemoryRetriever
    retrieval_limit: int = 5

    @classmethod
    def from_database_path(cls, database_path: str, retrieval_limit: int = 5) -> "MemoryManager":
        store = MemoryStore(database_path)
        return cls(
            store=store,
            retriever=MemoryRetriever(store),
            retrieval_limit=retrieval_limit,
        )

    def recall_context(self, user_input: str) -> str | None:
        memories = self.retriever.retrieve(user_input, limit=self.retrieval_limit)
        if not memories:
            return None

        lines = ["Relevant long-term memories:"]
        lines.extend(f"- [{memory.memory_type.value}] {memory.content}" for memory in memories)
        lines.append("")
        lines.append("Use these memories as private context. Answer naturally.")
        return "\n".join(lines)

    def remember_from_user_message(self, user_input: str) -> list[MemoryRecord]:
        candidates = extract_memory_candidates(user_input)
        return [self.store.add_candidate(candidate) for candidate in candidates]

    def close(self) -> None:
        self.store.close()
