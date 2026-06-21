"""Format graph query results into LLM context for JARVIS Phase 4.

Converts raw Neo4j relationship records into natural-language context
that gets injected into the system prompt alongside SQLite memories
and RAG document chunks.

Design decisions:
    - Output format uses bullet points grouped by relationship type,
      matching the style in ``memory_manager.recall_context()``.
    - The context is kept concise to avoid consuming too much of the
      LLM's context window.
    - Relationship labels are humanised: ``WORKS_ON`` → ``works on``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Human-readable labels for relationship types.
_RELATIONSHIP_DISPLAY: dict[str, str] = {
    "WORKS_ON": "works on",
    "USES": "uses",
    "LEARNS": "is learning",
    "INTERESTED_IN": "is interested in",
    "BELONGS_TO": "belongs to",
    "HAS_TASK": "has a task to",
    "KNOWS": "knows",
    "ASSIGNED": "was assigned",
    "HAS_DOCUMENT": "has document",
    "MENTIONS": "mentions",
    "ABOUT": "is about",
    "RELATED_TO": "is related to",
}


@dataclass(frozen=True)
class GraphContextBuilder:
    """Converts graph data into natural-language context for the LLM."""

    def build_context_from_relationships(
        self,
        entity_name: str,
        relationships: list[dict[str, Any]],
    ) -> str | None:
        """Format an entity's relationships as context text.

        Args:
            entity_name:   The name of the entity being queried.
            relationships: Records from ``GraphService.get_relationships()``.

        Returns:
            A formatted string, or None if there are no relationships.
        """
        if not relationships:
            return None

        lines = [f"Knowledge graph — relationships for {entity_name}:"]
        for record in relationships:
            rel = record["relationship"]
            neighbor = record["neighbor_name"]
            outgoing = record.get("outgoing", True)
            display_rel = _RELATIONSHIP_DISPLAY.get(rel, rel.lower().replace("_", " "))

            if outgoing:
                lines.append(f"  - {entity_name} {display_rel} {neighbor}")
            else:
                lines.append(f"  - {neighbor} {display_rel} {entity_name}")

        return "\n".join(lines)

    def build_context_from_user_world(
        self,
        user_context: list[dict[str, Any]],
    ) -> str | None:
        """Format the user's entire world model as context.

        Args:
            user_context: Records from ``GraphService.get_user_context()``.

        Returns:
            A formatted string, or None if the world model is empty.
        """
        if not user_context:
            return None

        # Group by relationship type for cleaner output.
        grouped: dict[str, list[str]] = {}
        for record in user_context:
            rel = record["relationship"]
            name = record["name"]
            display_rel = _RELATIONSHIP_DISPLAY.get(rel, rel.lower().replace("_", " "))
            grouped.setdefault(display_rel, []).append(name)

        lines = ["Knowledge graph — Atharva's world model:"]
        for display_rel, names in grouped.items():
            names_str = ", ".join(names)
            lines.append(f"  - {display_rel}: {names_str}")

        lines.append("")
        lines.append("Use these relationships as context. Answer naturally.")
        return "\n".join(lines)

    def build_context_from_connections(
        self,
        entity_a: str,
        entity_b: str,
        connections: list[dict[str, Any]],
    ) -> str | None:
        """Format shortest-path results between two entities.

        Args:
            entity_a:    Starting entity name.
            entity_b:    Ending entity name.
            connections: Records from ``GraphService.find_connections()``.

        Returns:
            A formatted string, or None if no path was found.
        """
        if not connections:
            return None

        lines = [f"Knowledge graph — connections between {entity_a} and {entity_b}:"]
        for connection in connections:
            nodes = connection["node_names"]
            rels = connection["rel_types"]
            path_parts: list[str] = []
            for i, node in enumerate(nodes):
                path_parts.append(node)
                if i < len(rels):
                    path_parts.append(f"—[{rels[i]}]→")
            lines.append("  " + " ".join(path_parts))

        return "\n".join(lines)
