"""Small Ollama HTTP client using only the Python standard library."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable
from urllib import error, request


class OllamaConnectionError(RuntimeError):
    """Raised when JARVIS cannot reach the local Ollama server."""


@dataclass(frozen=True)
class ChatMessage:
    """A chat message in Ollama's expected role/content format."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class OllamaClient:
    """Client for Ollama's local `/api/chat` endpoint."""

    host: str
    model: str
    timeout_seconds: int
    temperature: float
    context_window: int

    def chat(self, messages: Iterable[ChatMessage]) -> str:
        """Return a single assistant response from Ollama."""
        endpoint = f"{self.host.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.context_window,
            },
        }

        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise OllamaConnectionError(
                f"Ollama returned HTTP {exc.code}. Details: {details}"
            ) from exc
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaConnectionError(
                "I cannot reach Ollama at "
                f"{self.host}. Start Ollama, then run `ollama run {self.model}` "
                "once to download the local model. No cloud cavalry required."
            ) from exc

        message = data.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaConnectionError("Ollama responded, but no assistant text was returned.")

        return content.strip()
