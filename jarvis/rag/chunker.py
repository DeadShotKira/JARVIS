"""Document chunking for retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.rag.document_types import DocumentChunk, LoadedDocument


@dataclass(frozen=True)
class TextChunker:
    """Splits documents into overlapping retrieval chunks."""

    chunk_size: int = 900
    chunk_overlap: int = 150

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def chunk(self, document: LoadedDocument) -> list[DocumentChunk]:
        paragraphs = [paragraph.strip() for paragraph in document.text.split("\n\n") if paragraph.strip()]
        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            if len(paragraph) > self.chunk_size:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(self._split_long_text(paragraph))
                continue

            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                chunks.append(current.strip())
                current = self._overlap_tail(current, paragraph)

        if current:
            chunks.append(current.strip())

        return [
            DocumentChunk(
                id=f"{document.source_id}:{index}",
                text=chunk_text,
                metadata={
                    **document.metadata,
                    "source_id": document.source_id,
                    "chunk_id": index,
                    "chunk_count": len(chunks),
                },
            )
            for index, chunk_text in enumerate(chunks)
        ]

    def _split_long_text(self, text: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end].strip())
            if end == len(text):
                break
            start = max(0, end - self.chunk_overlap)
        return [chunk for chunk in chunks if chunk]

    def _overlap_tail(self, current: str, next_paragraph: str) -> str:
        if self.chunk_overlap == 0:
            return next_paragraph

        tail = current[-self.chunk_overlap:].strip()
        candidate = f"{tail}\n\n{next_paragraph}".strip() if tail else next_paragraph
        if len(candidate) <= self.chunk_size:
            return candidate
        return next_paragraph
