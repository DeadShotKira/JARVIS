"""Tiny YAML reader for JARVIS config files.

This intentionally supports only the simple subset used by config.yaml:
comments, blank lines, nested maps by two-space indentation, and scalar
strings/numbers/booleans. It keeps Phase 2 dependency-free for Raspberry Pi.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_simple_yaml(path: Path) -> dict[str, Any]:
    """Load a small YAML mapping from disk."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            raise ValueError(f"Invalid config line {line_number}: {raw_line!r}")

        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            raise ValueError(f"Invalid empty config key on line {line_number}")

        while stack and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        if raw_value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(raw_value)

    return root


def _parse_scalar(value: str) -> Any:
    stripped = value.strip()
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False

    if (stripped.startswith('"') and stripped.endswith('"')) or (
        stripped.startswith("'") and stripped.endswith("'")
    ):
        return stripped[1:-1]

    try:
        return int(stripped)
    except ValueError:
        pass

    try:
        return float(stripped)
    except ValueError:
        return stripped
