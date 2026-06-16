"""High-level RAG orchestration and knowledge management."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.rag.chunker import TextChunker
from jarvis.rag.context_builder import RagContextBuilder
from jarvis.rag.document_loader import DocumentLoader
from jarvis.rag.document_types import SearchResult
from jarvis.rag.embedder import Embedder, create_embedder
from jarvis.rag.retriever import DocumentRetriever
from jarvis.rag.vector_store import VectorStore, create_vector_store


@dataclass
class RagManager:
    """Coordinates ingestion, vector search, and context building."""

    loader: DocumentLoader
    chunker: TextChunker
    embedder: Embedder
    vector_store: VectorStore
    retriever: DocumentRetriever
    context_builder: RagContextBuilder
    uploads_path: Path
    processed_path: Path

    @classmethod
    def from_settings(cls, settings: Any) -> "RagManager":
        uploads_path = settings.rag_uploads_path
        processed_path = settings.rag_processed_path
        indexes_path = settings.rag_indexes_path
        uploads_path.mkdir(parents=True, exist_ok=True)
        processed_path.mkdir(parents=True, exist_ok=True)
        indexes_path.mkdir(parents=True, exist_ok=True)

        embedder = create_embedder(
            provider=settings.rag_embedding_provider,
            model_name=settings.rag_embedding_model,
            dimensions=settings.rag_embedding_dimensions,
        )
        vector_store = create_vector_store(
            backend=settings.rag_vector_backend,
            indexes_path=indexes_path,
            collection_name=settings.rag_collection_name,
        )
        retriever = DocumentRetriever(
            embedder=embedder,
            vector_store=vector_store,
            top_k=settings.rag_top_k,
            similarity_threshold=settings.rag_similarity_threshold,
        )
        return cls(
            loader=DocumentLoader(),
            chunker=TextChunker(
                chunk_size=settings.rag_chunk_size,
                chunk_overlap=settings.rag_chunk_overlap,
            ),
            embedder=embedder,
            vector_store=vector_store,
            retriever=retriever,
            context_builder=RagContextBuilder(),
            uploads_path=uploads_path,
            processed_path=processed_path,
        )

    def ingest_document(self, path: Path | str) -> dict[str, Any]:
        """Load, chunk, embed, and index one document."""
        source_path = Path(path).expanduser().resolve()
        uploaded_path = self._copy_to_uploads(source_path)
        document = self.loader.load(uploaded_path)
        chunks = self.chunker.chunk(document)
        embeddings = self.embedder.embed_texts([chunk.text for chunk in chunks])
        self.vector_store.update_document(chunks, embeddings)

        manifest = {
            "source_id": document.source_id,
            "filename": document.metadata["filename"],
            "document_type": document.metadata["document_type"],
            "source_path": document.metadata["source_path"],
            "chunk_count": len(chunks),
            "embedding_model": self.embedder.model_name,
        }
        self._write_manifest(manifest)
        return manifest

    def retrieve(self, query: str) -> list[SearchResult]:
        return self.retriever.retrieve(query)

    def build_context(self, user_query: str, memory_context: str | None = None) -> str | None:
        document_results = self.retrieve(user_query)
        return self.context_builder.build(
            user_query=user_query,
            memory_context=memory_context,
            document_results=document_results,
        )

    def list_documents(self) -> list[dict[str, Any]]:
        by_source_id = {str(item.get("source_id", "")): item for item in self.vector_store.list_documents()}
        for manifest in self._read_manifests():
            by_source_id.setdefault(str(manifest.get("source_id", "")), manifest)
        return sorted(by_source_id.values(), key=lambda item: str(item.get("filename", "")))

    def remove_document(self, source_id_or_filename: str) -> bool:
        target = self._find_document(source_id_or_filename)
        if not target:
            return False
        source_id = str(target["source_id"])
        self.vector_store.delete_document(source_id)
        manifest_path = self._manifest_path(source_id)
        if manifest_path.exists():
            manifest_path.unlink()
        return True

    def rebuild_index(self) -> list[dict[str, Any]]:
        for document in self.list_documents():
            source_id = str(document.get("source_id", ""))
            if source_id:
                self.vector_store.delete_document(source_id)
        for manifest_path in self.processed_path.glob("*.json"):
            manifest_path.unlink()

        results: list[dict[str, Any]] = []
        for path in sorted(self.uploads_path.iterdir()):
            if path.is_file():
                results.append(self.ingest_document(path))
        return results

    def document_metadata(self, source_id_or_filename: str) -> dict[str, Any] | None:
        return self._find_document(source_id_or_filename)

    def _copy_to_uploads(self, source_path: Path) -> Path:
        if source_path.parent == self.uploads_path.resolve():
            return source_path
        self.uploads_path.mkdir(parents=True, exist_ok=True)
        destination = self.uploads_path / source_path.name
        if destination.resolve() == source_path:
            return destination
        shutil.copy2(source_path, destination)
        return destination.resolve()

    def _find_document(self, source_id_or_filename: str) -> dict[str, Any] | None:
        needle = source_id_or_filename.strip().lower()
        for document in self.list_documents():
            source_id = str(document.get("source_id", "")).lower()
            filename = str(document.get("filename", "")).lower()
            if needle in {source_id, filename}:
                return document
        return None

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self._manifest_path(str(manifest["source_id"])).write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _read_manifests(self) -> list[dict[str, Any]]:
        manifests: list[dict[str, Any]] = []
        if not self.processed_path.exists():
            return manifests
        for path in sorted(self.processed_path.glob("*.json")):
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
        return manifests

    def _manifest_path(self, source_id: str) -> Path:
        return self.processed_path / f"{source_id}.json"
