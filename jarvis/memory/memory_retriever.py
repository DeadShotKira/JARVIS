"""Relevant memory retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.memory.memory_store import MemoryRecord, MemoryStore
from jarvis.memory.memory_types import MemoryType
from jarvis.memory.memory_utils import infer_requested_memory_types, tokenize


@dataclass
class MemoryRetriever:
    """Retrieves memories with simple lexical scoring.

    This is not RAG and not vector search yet. It is a transparent Phase 2
    bridge that can later be replaced by embeddings.
    """

    store: MemoryStore

    def retrieve(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        query_terms = tokenize(query)
        requested_types = infer_requested_memory_types(query)

        scored: list[tuple[float, MemoryRecord]] = []
        for record in self.store.list_memories():
            score = self._score(record, query_terms, requested_types)
            if score > 0:
                scored.append((score, record))

        scored.sort(key=lambda item: (item[0], item[1].importance, item[1].updated_at), reverse=True)
        selected = [record for _, record in scored[:limit]]
        self.store.touch_memories([record.id for record in selected])
        return selected

    def _score(
        self,
        record: MemoryRecord,
        query_terms: set[str],
        requested_types: set[MemoryType],
    ) -> float:
        memory_terms = tokenize(record.content)
        overlap = len(query_terms.intersection(memory_terms))
        score = float(overlap)

        if record.memory_type in requested_types:
            score += 5.0

        if query_terms and any(term in memory_terms for term in query_terms):
            score += record.importance / 10

        return score
