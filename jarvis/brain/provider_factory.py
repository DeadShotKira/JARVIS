"""Model provider factory."""

from __future__ import annotations

from jarvis.brain.model_client import ModelClient
from jarvis.brain.ollama_client import OllamaClient
from jarvis.config.settings import Settings


def create_model_client(settings: Settings) -> ModelClient:
    """Create the configured model backend."""
    if settings.provider == "ollama":
        return OllamaClient(
            host=settings.ollama_host,
            model=settings.active_model,
            timeout_seconds=settings.request_timeout_seconds,
            temperature=settings.temperature,
            context_window=settings.context_window,
        )

    raise ValueError(
        "Unsupported model provider. "
        f"Configured provider: {settings.provider}. "
        "Phase 2 ships with Ollama; add a provider client behind this factory later."
    )
