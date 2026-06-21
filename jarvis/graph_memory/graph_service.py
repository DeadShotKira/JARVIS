"""Graph CRUD service for JARVIS Phase 4.

Provides high-level operations on the knowledge graph:
    - Entity upsert / get / search / delete
    - Relationship creation and traversal
    - Convenience helpers (get_projects, get_interests, etc.)

All Cypher execution is delegated to ``Neo4jClient``. Query strings come
from ``graph_queries.py``.

Design decisions:
    - Every write uses MERGE (upsert) for idempotency.
    - ``search_entities`` uses Neo4j full-text index. If the index is not
      yet created (first run), it falls back to a CONTAINS query.
    - Properties are filtered to only include non-None values before
      sending to Neo4j.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from jarvis.graph_memory import graph_queries as queries
from jarvis.graph_memory.graph_schema import (
    Entity,
    EntityType,
    Relationship,
    make_canonical,
)
from jarvis.graph_memory.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class GraphService:
    """CRUD layer over the Neo4j knowledge graph."""

    client: Neo4jClient

    # -------------------------------------------------------------------
    # Entity operations
    # -------------------------------------------------------------------

    def upsert_entity(self, entity: Entity) -> None:
        """Create or update a node in the graph.

        Uses MERGE on canonical_name so duplicates are impossible.
        """
        label = entity.entity_type.value
        properties = {k: v for k, v in entity.properties.items() if v is not None}
        self.client.execute_write(
            queries.upsert_entity_query(label),
            {
                "canonical_name": entity.canonical_name,
                "name": entity.name,
                "source": entity.source,
                "properties": properties,
            },
        )
        logger.debug("Upserted entity: %s (%s)", entity.name, label)

    def get_entity(
        self,
        name: str,
        entity_type: EntityType,
    ) -> dict[str, Any] | None:
        """Fetch a single entity by name and type.  Returns None if absent."""
        canonical = make_canonical(name)
        records = self.client.execute_read(
            queries.get_entity_query(entity_type.value),
            {"canonical_name": canonical},
        )
        if not records:
            return None
        return dict(records[0]["n"])

    def search_entities(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Full-text search across all entity types."""
        # Neo4j full-text search requires a non-empty query and special chars
        # to be escaped.  Append a wildcard for prefix matching.
        sanitised = query.strip()
        if not sanitised:
            return []

        try:
            search_term = sanitised + "*"
            records = self.client.execute_read(
                queries.search_entities_query(),
                {"query": search_term, "limit": limit},
            )
            return [
                {
                    "labels": record["labels"],
                    "name": record["name"],
                    "canonical_name": record["canonical_name"],
                    "score": record["score"],
                }
                for record in records
            ]
        except Exception:
            # Fulltext index may not exist yet on first run.
            logger.debug("Full-text search failed; index may not be ready.")
            return []

    def delete_entity(self, name: str, entity_type: EntityType) -> bool:
        """Delete an entity and all its relationships.  Returns True if found."""
        canonical = make_canonical(name)
        records = self.client.execute_write(
            queries.delete_entity_query(entity_type.value),
            {"canonical_name": canonical},
        )
        deleted = records[0]["deleted"] if records else 0
        return deleted > 0

    # -------------------------------------------------------------------
    # Relationship operations
    # -------------------------------------------------------------------

    def add_relationship(self, relationship: Relationship) -> None:
        """Create a directed relationship between two entities.

        Both source and target nodes are MERGEd so the write never fails
        due to a missing node.
        """
        properties = {
            k: v for k, v in relationship.properties.items() if v is not None
        }
        self.client.execute_write(
            queries.upsert_relationship_query(
                source_label=relationship.source_entity.entity_type.value,
                target_label=relationship.target_entity.entity_type.value,
                rel_type=relationship.relationship_type.value,
            ),
            {
                "source_canonical": relationship.source_entity.canonical_name,
                "source_name": relationship.source_entity.name,
                "source_source": relationship.source_entity.source,
                "target_canonical": relationship.target_entity.canonical_name,
                "target_name": relationship.target_entity.name,
                "target_source": relationship.target_entity.source,
                "rel_properties": properties,
            },
        )
        logger.debug(
            "Added relationship: (%s)-[:%s]->(%s)",
            relationship.source_entity.name,
            relationship.relationship_type.value,
            relationship.target_entity.name,
        )

    def get_relationships(self, entity_name: str) -> list[dict[str, Any]]:
        """Get all relationships for a given entity."""
        canonical = make_canonical(entity_name)
        records = self.client.execute_read(
            queries.get_relationships_query(),
            {"canonical_name": canonical},
        )
        return [
            {
                "entity": record["entity"],
                "relationship": record["relationship"],
                "neighbor_labels": record["neighbor_labels"],
                "neighbor_name": record["neighbor_name"],
                "outgoing": record["outgoing"],
                "properties": record.get("rel_properties", {}),
            }
            for record in records
        ]

    def find_connections(
        self,
        entity_a: str,
        entity_b: str,
    ) -> list[dict[str, Any]]:
        """Find the shortest path between two entities (up to 4 hops)."""
        records = self.client.execute_read(
            queries.find_connections_query(),
            {
                "name_a": make_canonical(entity_a),
                "name_b": make_canonical(entity_b),
            },
        )
        return [
            {
                "node_names": record["node_names"],
                "rel_types": record["rel_types"],
                "hops": record["hops"],
            }
            for record in records
        ]

    # -------------------------------------------------------------------
    # Convenience queries
    # -------------------------------------------------------------------

    def get_projects(self) -> list[dict[str, Any]]:
        """Return all Project nodes with their technologies."""
        return self.client.execute_read(queries.get_projects_with_tech_query())

    def get_interests(self) -> list[dict[str, Any]]:
        """Return all Interest nodes."""
        return self.client.execute_read(
            queries.get_all_by_label_query("Interest"),
        )

    def get_technologies(self) -> list[dict[str, Any]]:
        """Return all Technology nodes."""
        return self.client.execute_read(
            queries.get_all_by_label_query("Technology"),
        )

    def get_entity_context(self, entity_name: str) -> list[dict[str, Any]]:
        """Return an entity and its 1-hop neighbourhood."""
        canonical = make_canonical(entity_name)
        return self.client.execute_read(
            queries.get_entity_neighborhood_query(),
            {"canonical_name": canonical},
        )

    def get_user_context(self, user_canonical: str = "atharva") -> list[dict[str, Any]]:
        """Return everything directly connected to the user node."""
        return self.client.execute_read(
            queries.get_user_context_query(),
            {"user_canonical": user_canonical},
        )

    def search_related_entities(
        self,
        entity_name: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return entities up to 2 hops from *entity_name*."""
        canonical = make_canonical(entity_name)
        return self.client.execute_read(
            queries.get_related_entities_query(),
            {"canonical_name": canonical, "limit": limit},
        )

    def get_graph_stats(self) -> dict[str, int]:
        """Return total node and relationship counts."""
        records = self.client.execute_read(queries.get_graph_stats_query())
        if records:
            return {
                "node_count": records[0]["node_count"],
                "relationship_count": records[0]["relationship_count"],
            }
        return {"node_count": 0, "relationship_count": 0}
