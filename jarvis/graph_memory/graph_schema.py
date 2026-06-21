"""Graph schema definitions for JARVIS Phase 4 Knowledge Graph.

Defines the entity types, relationship types, and data structures used
throughout the graph_memory module. Also provides schema initialization
(constraints and indexes) for Neo4j.

Design decisions:
    - Each entity type has a canonical_name (lowercase) for deduplication
      and a display name preserving original casing.
    - Relationship types are a closed enum so extraction code and query code
      stay in sync.
    - All timestamps are ISO-8601 strings matching the convention in
      jarvis.memory.memory_store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Entity types
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    """Supported node labels in the knowledge graph."""

    PERSON = "Person"
    PROJECT = "Project"
    TECHNOLOGY = "Technology"
    ORGANIZATION = "Organization"
    INTEREST = "Interest"
    DOCUMENT = "Document"
    TASK = "Task"


# ---------------------------------------------------------------------------
# Relationship types
# ---------------------------------------------------------------------------

class RelationshipType(str, Enum):
    """Supported edge types in the knowledge graph."""

    # Person relationships
    WORKS_ON = "WORKS_ON"
    USES = "USES"
    LEARNS = "LEARNS"
    INTERESTED_IN = "INTERESTED_IN"
    BELONGS_TO = "BELONGS_TO"
    HAS_TASK = "HAS_TASK"
    KNOWS = "KNOWS"

    # Project relationships
    HAS_DOCUMENT = "HAS_DOCUMENT"

    # Organization relationships
    ASSIGNED = "ASSIGNED"

    # Document relationships
    MENTIONS = "MENTIONS"
    ABOUT = "ABOUT"

    # Technology relationships
    RELATED_TO = "RELATED_TO"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Entity:
    """A named entity extracted from text, ready for graph insertion.

    Attributes:
        name:            Display name preserving original casing.
        canonical_name:  Lowercase key used for MERGE deduplication.
        entity_type:     The graph node label.
        properties:      Optional extra properties (status, category, etc.).
        source:          Where this entity was detected (conversation, rag, manual).
    """

    name: str
    canonical_name: str
    entity_type: EntityType
    properties: dict[str, Any] = field(default_factory=dict)
    source: str = "conversation"


@dataclass(frozen=True)
class Relationship:
    """A directed edge between two entities.

    Attributes:
        source_entity:  The starting node (name, type).
        target_entity:  The ending node (name, type).
        relationship_type:  The edge label.
        properties:     Optional edge properties (since, role, etc.).
    """

    source_entity: Entity
    target_entity: Entity
    relationship_type: RelationshipType
    properties: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Schema initialisation queries
# ---------------------------------------------------------------------------

# Uniqueness constraints — one per entity type on canonical_name.
# These also serve as implicit indexes on canonical_name.
_CONSTRAINT_QUERIES: list[str] = [
    f"""
    CREATE CONSTRAINT {entity_type.value.lower()}_unique IF NOT EXISTS
    FOR (n:{entity_type.value}) REQUIRE n.canonical_name IS UNIQUE
    """
    for entity_type in EntityType
]

# Full-text index for natural language entity search across all node types.
_FULLTEXT_INDEX_QUERY = """
CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
FOR (n:Person|Project|Technology|Organization|Interest|Document|Task)
ON EACH [n.name, n.canonical_name]
"""


def get_schema_queries() -> list[str]:
    """Return the Cypher statements needed to initialise the graph schema.

    Safe to run repeatedly — every statement uses IF NOT EXISTS.
    """
    return _CONSTRAINT_QUERIES + [_FULLTEXT_INDEX_QUERY]


def get_seed_user_query() -> str:
    """Return a MERGE query that ensures the primary user node exists."""
    return """
    MERGE (p:Person {canonical_name: "atharva"})
    ON CREATE SET
        p.name = "Atharva",
        p.created_at = datetime(),
        p.source = "system"
    RETURN p
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_canonical(name: str) -> str:
    """Normalise a name for deduplication.

    Strips whitespace, lowercases, and collapses internal spaces.
    """
    return " ".join(name.strip().lower().split())


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp matching the project convention."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
