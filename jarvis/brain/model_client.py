"""Model client interface."""

from __future__ import annotations

from typing import Iterable, Protocol

from jarvis.brain.ollama_client import ChatMessage


class ModelClient(Protocol):
    """A chat model backend used by JarvisAssistant."""

    def chat(self, messages: Iterable[ChatMessage]) -> str:
        """Return a single assistant response."""
