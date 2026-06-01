"""Prompt file loading."""

from __future__ import annotations

from pathlib import Path


def load_personality_prompt(path: Path) -> str:
    """Load the assistant personality prompt from Markdown."""
    if not path.exists():
        raise FileNotFoundError(f"Personality prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()
