"""Document loading for local knowledge files."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from jarvis.rag.document_types import LoadedDocument


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


class DocumentLoadError(RuntimeError):
    """Raised when a document cannot be loaded."""


class DocumentLoader:
    """Extracts clean text and metadata from local documents."""

    def load(self, path: Path | str) -> LoadedDocument:
        resolved_path = Path(path).expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Document not found: {resolved_path}")

        extension = resolved_path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise DocumentLoadError(
                f"Unsupported document type: {extension}. "
                "Supported types: PDF, TXT, Markdown."
            )

        if extension == ".pdf":
            text = self._load_pdf(resolved_path)
        else:
            text = resolved_path.read_text(encoding="utf-8", errors="replace")

        cleaned_text = clean_text(text)
        if not cleaned_text:
            raise DocumentLoadError(f"No readable text found in {resolved_path.name}")

        stat = resolved_path.stat()
        metadata = {
            "filename": resolved_path.name,
            "source_path": str(resolved_path),
            "document_type": extension.lstrip("."),
            "created_date": datetime.fromtimestamp(stat.st_ctime, timezone.utc).isoformat(
                timespec="seconds"
            ),
            "modified_date": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(
                timespec="seconds"
            ),
        }
        source_id = stable_source_id(resolved_path)
        return LoadedDocument(source_id=source_id, text=cleaned_text, metadata=metadata)

    def _load_pdf(self, path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise DocumentLoadError(
                "PDF loading requires the optional `pypdf` package. "
                "Install project requirements, then try again."
            ) from exc

        reader = PdfReader(str(path))
        pages: list[str] = []
        for page_number, page in enumerate(reader.pages, 1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"\n[Page {page_number}]\n{page_text}")
        return "\n".join(pages)


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def stable_source_id(path: Path) -> str:
    """Create a stable id from path and file metadata."""
    stat = path.stat()
    fingerprint = f"{path.resolve()}::{stat.st_size}::{int(stat.st_mtime)}"
    return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()
