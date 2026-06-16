"""Build model context from memory and document retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.rag.document_types import SearchResult


@dataclass(frozen=True)
class RagContextBuilder:
    """Combines memories, retrieved documents, and the user question."""

    def build(
        self,
        user_query: str,
        memory_context: str | None = None,
        document_results: list[SearchResult] | None = None,
    ) -> str | None:
        sections: list[str] = []

        if memory_context:
            sections.append(memory_context.strip())

        if document_results:
            lines = ["Relevant documents:"]
            for index, result in enumerate(document_results, 1):
                filename = result.chunk.metadata.get("filename", "unknown source")
                chunk_id = result.chunk.metadata.get("chunk_id", "?")
                lines.append(
                    f"[{index}] Source: {filename} | Chunk: {chunk_id} | "
                    f"Similarity: {result.score:.3f}"
                )
                lines.append(result.chunk.text.strip())
                lines.append("")
            lines.append("When using document facts, mention the source filename naturally.")
            sections.append("\n".join(lines).strip())

        if not sections:
            return None

        sections.append(f"User question:\n{user_query.strip()}")
        return "\n\n".join(sections)
