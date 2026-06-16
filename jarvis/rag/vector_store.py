"""Vector database abstraction for document chunks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from jarvis.rag.document_types import DocumentChunk, SearchResult


class VectorStore(Protocol):
    """Common vector backend interface for retrieval code."""

    def add_documents(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Add document chunks and their embeddings."""

    def delete_document(self, source_id: str) -> None:
        """Delete all chunks for a source document."""

    def update_document(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Replace a document's chunks."""

    def search(self, embedding: list[float], top_k: int) -> list[SearchResult]:
        """Return nearest chunks for a query embedding."""

    def list_documents(self) -> list[dict[str, Any]]:
        """Return source document metadata known to the vector store."""


@dataclass
class InMemoryVectorStore:
    """Small vector store useful for tests and local debugging."""

    _items: dict[str, tuple[DocumentChunk, list[float]]] | None = None

    def __post_init__(self) -> None:
        if self._items is None:
            self._items = {}

    def add_documents(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        _validate_lengths(chunks, embeddings)
        assert self._items is not None
        for chunk, embedding in zip(chunks, embeddings):
            self._items[chunk.id] = (chunk, embedding)

    def delete_document(self, source_id: str) -> None:
        assert self._items is not None
        for chunk_id, (chunk, _) in list(self._items.items()):
            if chunk.metadata.get("source_id") == source_id:
                del self._items[chunk_id]

    def update_document(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if chunks:
            source_id = str(chunks[0].metadata["source_id"])
            self.delete_document(source_id)
        self.add_documents(chunks, embeddings)

    def search(self, embedding: list[float], top_k: int) -> list[SearchResult]:
        assert self._items is not None
        scored = [
            SearchResult(chunk=chunk, score=_cosine_similarity(embedding, stored_embedding))
            for chunk, stored_embedding in self._items.values()
        ]
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]

    def list_documents(self) -> list[dict[str, Any]]:
        assert self._items is not None
        documents: dict[str, dict[str, Any]] = {}
        for chunk, _ in self._items.values():
            source_id = str(chunk.metadata.get("source_id", ""))
            documents.setdefault(
                source_id,
                {
                    "source_id": source_id,
                    "filename": chunk.metadata.get("filename", ""),
                    "document_type": chunk.metadata.get("document_type", ""),
                    "chunk_count": chunk.metadata.get("chunk_count", 0),
                    "source_path": chunk.metadata.get("source_path", ""),
                },
            )
        return sorted(documents.values(), key=lambda item: str(item.get("filename", "")))


class ChromaVectorStore:
    """ChromaDB-backed vector store.

    ChromaDB is the primary Phase 3 backend because collections are easy to
    inspect, reset, and debug while learning RAG.
    """

    def __init__(self, persist_path: Path | str, collection_name: str):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "ChromaDB is configured as the vector backend, but `chromadb` is not installed."
            ) from exc

        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        _validate_lengths(chunks, embeddings)
        if not chunks:
            return
        self._collection.add(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[_sanitize_metadata(chunk.metadata) for chunk in chunks],
        )

    def delete_document(self, source_id: str) -> None:
        self._collection.delete(where={"source_id": source_id})

    def update_document(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if chunks:
            self.delete_document(str(chunks[0].metadata["source_id"]))
        self.add_documents(chunks, embeddings)

    def search(self, embedding: list[float], top_k: int) -> list[SearchResult]:
        if top_k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        search_results: list[SearchResult] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            score = 1.0 - float(distance)
            search_results.append(
                SearchResult(
                    chunk=DocumentChunk(id=str(chunk_id), text=str(text), metadata=dict(metadata or {})),
                    score=score,
                )
            )
        return search_results

    def list_documents(self) -> list[dict[str, Any]]:
        data = self._collection.get(include=["metadatas"])
        documents: dict[str, dict[str, Any]] = {}
        for metadata in data.get("metadatas", []):
            metadata = dict(metadata or {})
            source_id = str(metadata.get("source_id", ""))
            documents.setdefault(
                source_id,
                {
                    "source_id": source_id,
                    "filename": metadata.get("filename", ""),
                    "document_type": metadata.get("document_type", ""),
                    "chunk_count": metadata.get("chunk_count", 0),
                    "source_path": metadata.get("source_path", ""),
                },
            )
        return sorted(documents.values(), key=lambda item: str(item.get("filename", "")))


class FaissVectorStore:
    """FAISS adapter behind the same VectorStore interface."""

    def __init__(self, index_path: Path | str, metadata_path: Path | str):
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("FAISS backend requires `faiss-cpu` and `numpy`.") from exc

        self._faiss = faiss
        self._np = np
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._chunks: list[DocumentChunk] = []
        self._embeddings: list[list[float]] = []
        self._load()

    def add_documents(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        _validate_lengths(chunks, embeddings)
        self._chunks.extend(chunks)
        self._embeddings.extend(embeddings)
        self._persist()

    def delete_document(self, source_id: str) -> None:
        retained = [
            (chunk, embedding)
            for chunk, embedding in zip(self._chunks, self._embeddings)
            if chunk.metadata.get("source_id") != source_id
        ]
        self._chunks = [chunk for chunk, _ in retained]
        self._embeddings = [embedding for _, embedding in retained]
        self._persist()

    def update_document(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if chunks:
            self.delete_document(str(chunks[0].metadata["source_id"]))
        self.add_documents(chunks, embeddings)

    def search(self, embedding: list[float], top_k: int) -> list[SearchResult]:
        if not self._embeddings or top_k <= 0:
            return []
        index = self._build_index()
        query = self._np.array([embedding], dtype="float32")
        self._faiss.normalize_L2(query)
        scores, indices = index.search(query, min(top_k, len(self._chunks)))
        results: list[SearchResult] = []
        for score, index_id in zip(scores[0], indices[0]):
            if index_id < 0:
                continue
            results.append(SearchResult(chunk=self._chunks[int(index_id)], score=float(score)))
        return results

    def list_documents(self) -> list[dict[str, Any]]:
        documents: dict[str, dict[str, Any]] = {}
        for chunk in self._chunks:
            source_id = str(chunk.metadata.get("source_id", ""))
            documents.setdefault(
                source_id,
                {
                    "source_id": source_id,
                    "filename": chunk.metadata.get("filename", ""),
                    "document_type": chunk.metadata.get("document_type", ""),
                    "chunk_count": chunk.metadata.get("chunk_count", 0),
                    "source_path": chunk.metadata.get("source_path", ""),
                },
            )
        return sorted(documents.values(), key=lambda item: str(item.get("filename", "")))

    def _build_index(self):
        dimension = len(self._embeddings[0])
        index = self._faiss.IndexFlatIP(dimension)
        matrix = self._np.array(self._embeddings, dtype="float32")
        self._faiss.normalize_L2(matrix)
        index.add(matrix)
        return index

    def _persist(self) -> None:
        if self._embeddings:
            self._faiss.write_index(self._build_index(), str(self.index_path))
        payload = [
            {"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata, "embedding": embedding}
            for chunk, embedding in zip(self._chunks, self._embeddings)
        ]
        self.metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _load(self) -> None:
        if not self.metadata_path.exists():
            return
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self._chunks = [
            DocumentChunk(id=str(item["id"]), text=str(item["text"]), metadata=dict(item["metadata"]))
            for item in payload
        ]
        self._embeddings = [list(map(float, item["embedding"])) for item in payload]


def create_vector_store(
    backend: str,
    indexes_path: Path,
    collection_name: str,
) -> VectorStore:
    """Create the configured vector backend."""
    normalized = backend.strip().lower()
    if normalized == "chromadb":
        return ChromaVectorStore(indexes_path / "chromadb", collection_name)
    if normalized == "faiss":
        return FaissVectorStore(
            index_path=indexes_path / "faiss" / f"{collection_name}.index",
            metadata_path=indexes_path / "faiss" / f"{collection_name}.json",
        )
    if normalized == "memory":
        return InMemoryVectorStore()
    raise ValueError(f"Unsupported vector backend: {backend}")


def _validate_lengths(chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match")
    numerator = sum(a * b for a, b in zip(left, right))
    left_mag = math.sqrt(sum(a * a for a in left))
    right_mag = math.sqrt(sum(b * b for b in right))
    if left_mag == 0 or right_mag == 0:
        return 0.0
    return numerator / (left_mag * right_mag)


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif value is None:
            sanitized[key] = ""
        else:
            sanitized[key] = json.dumps(value, sort_keys=True)
    return sanitized
