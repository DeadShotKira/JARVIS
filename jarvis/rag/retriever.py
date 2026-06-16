"""Document retrieval over a vector store."""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.rag.document_types import SearchResult
from jarvis.rag.embedder import Embedder
from jarvis.rag.vector_store import VectorStore


@dataclass
class DocumentRetriever:
    """Embeds a question and finds similar document chunks."""

    embedder: Embedder
    vector_store: VectorStore
    top_k: int = 4
    similarity_threshold: float = 0.25

    def retrieve(self, query: str) -> list[SearchResult]:
        query_embedding = self.embedder.embed_text(query)
        results = self.vector_store.search(query_embedding, top_k=self.top_k)
        return [result for result in results if result.score >= self.similarity_threshold]
