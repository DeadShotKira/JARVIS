"""Local embedding providers."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol


class Embedder(Protocol):
    """Converts text into vectors."""

    model_name: str
    dimensions: int

    def embed_text(self, text: str) -> list[float]:
        """Embed one text string."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed many text strings."""


@dataclass
class HashingEmbedder:
    """Dependency-free local embedder for transparent development and tests.

    This is not as semantically rich as a transformer embedding model, but it
    is local, deterministic, inspectable, and follows the same interface.
    """

    model_name: str
    dimensions: int = 384

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector
        return [value / magnitude for value in vector]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class SentenceTransformerEmbedder:
    """Local open-source transformer embeddings via sentence-transformers."""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for the configured embedding provider."
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]


def create_embedder(provider: str, model_name: str, dimensions: int) -> Embedder:
    """Create a configured local embedder."""
    normalized = provider.strip().lower()
    if normalized == "hashing":
        return HashingEmbedder(model_name=model_name, dimensions=dimensions)
    if normalized in {"sentence_transformers", "sentence-transformers"}:
        return SentenceTransformerEmbedder(model_name=model_name)
    raise ValueError(f"Unsupported embedding provider: {provider}")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_+-]*", text.lower())
