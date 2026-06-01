"""Runtime settings for JARVIS v0.2."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.config.simple_yaml import load_simple_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Configuration loaded from config.yaml."""

    active_model: str
    provider: str
    ollama_host: str
    request_timeout_seconds: int
    runtime_max_messages: int
    memory_database_path: Path
    memory_retrieval_limit: int
    temperature: float
    context_window: int
    personality_path: Path


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from config.yaml.

    Set JARVIS_CONFIG_PATH to point at a different config file for Docker,
    Raspberry Pi, or tests.
    """
    resolved_config_path = config_path or Path(
        os.getenv("JARVIS_CONFIG_PATH", str(PROJECT_ROOT / "config.yaml"))
    )
    resolved_config_path = resolved_config_path.expanduser().resolve()
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"JARVIS config file not found: {resolved_config_path}")

    config = load_simple_yaml(resolved_config_path)
    config_base = resolved_config_path.parent

    return Settings(
        active_model=_required_str(config, "active_model"),
        provider=_required_str(config, "provider"),
        ollama_host=_required_str(config, "ollama.host"),
        request_timeout_seconds=_required_int(config, "ollama.request_timeout_seconds"),
        runtime_max_messages=_required_int(config, "memory.runtime_max_messages"),
        memory_database_path=_resolve_path(
            config_base,
            _required_str(config, "memory.database_path"),
        ),
        memory_retrieval_limit=_required_int(config, "memory.retrieval_limit"),
        temperature=_required_float(config, "generation.temperature"),
        context_window=_required_int(config, "generation.context_window"),
        personality_path=_resolve_path(
            config_base,
            _required_str(config, "prompts.personality_path"),
        ),
    )


def _required(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Missing required config value: {dotted_key}")
        current = current[part]
    return current


def _required_str(config: dict[str, Any], dotted_key: str) -> str:
    value = _required(config, dotted_key)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{dotted_key} must be a non-empty string")
    return value.strip()


def _required_int(config: dict[str, Any], dotted_key: str) -> int:
    value = _required(config, dotted_key)
    if not isinstance(value, int):
        raise TypeError(f"{dotted_key} must be an integer")
    return value


def _required_float(config: dict[str, Any], dotted_key: str) -> float:
    value = _required(config, dotted_key)
    if not isinstance(value, (float, int)):
        raise TypeError(f"{dotted_key} must be a number")
    return float(value)


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()
