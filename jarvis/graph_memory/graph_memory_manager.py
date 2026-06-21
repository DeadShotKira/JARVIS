"""High-level Knowledge Graph orchestrator for JARVIS Phase 4.

Coordinates entity extraction, relationship extraction, graph storage,
and context building. This is the graph-layer counterpart of
``jarvis.memory.memory_manager.MemoryManager``.

Usage:
    manager = GraphMemoryManager.from_settings(settings)
    if manager:
        manager.extract_and_store("I am building Jarvis using Neo4j")
        context = manager.graph_context("What am I working on?")

Design decisions:
    - ``from_settings`` returns None if graph is disabled or Neo4j is
      unreachable, so the rest of JARVIS keeps running. This mirrors
      how ``rag_manager`` is None when RAG is disabled.
    - Extraction runs the entity extractor first, then the relationship
      extractor receives the extracted entities as input.
    - Context building queries the user's world model for general
      questions and does targeted entity lookups for specific queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from jarvis.graph_memory.entity_extractor import (
    EntityExtractor,
    RuleBasedEntityExtractor,
)
from jarvis.graph_memory.graph_context_builder import GraphContextBuilder
from jarvis.graph_memory.graph_schema import (
    Entity,
    EntityType,
    get_schema_queries,
    get_seed_user_query,
    make_canonical,
)
from jarvis.graph_memory.graph_service import GraphService
from jarvis.graph_memory.neo4j_client import Neo4jClient, Neo4jConnectionError
from jarvis.graph_memory.relationship_extractor import (
    RelationshipExtractor,
    RuleBasedRelationshipExtractor,
)

logger = logging.getLogger(__name__)


@dataclass
class GraphMemoryManager:
    """Orchestrates entity extraction, graph storage, and context building.

    Attributes:
        client:                 Neo4j connection wrapper.
        service:                Graph CRUD layer.
        entity_extractor:       Extracts entities from text.
        relationship_extractor: Extracts relationships from text + entities.
        context_builder:        Formats graph data for the LLM.
    """

    client: Neo4jClient
    service: GraphService
    entity_extractor: EntityExtractor
    relationship_extractor: RelationshipExtractor
    context_builder: GraphContextBuilder

    # -------------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Any) -> "GraphMemoryManager | None":
        """Create the graph manager from application settings.

        Returns None if graph is disabled in configuration or Neo4j is
        not reachable. In that case JARVIS continues without graph features
        (graceful degradation).
        """
        if not getattr(settings, "graph_enabled", False):
            logger.info("Knowledge graph is disabled in configuration.")
            return None

        client = Neo4jClient(
            uri=settings.graph_uri,
            username=settings.graph_username,
            password=settings.graph_password,
            database=settings.graph_database,
        )

        try:
            client.connect()
        except Neo4jConnectionError as exc:
            logger.warning(
                "Knowledge graph unavailable — continuing without graph features. "
                "Reason: %s",
                exc,
            )
            return None

        # Initialise schema (constraints + indexes). Safe to run repeatedly.
        try:
            for query in get_schema_queries():
                client.execute_write(query)
            client.execute_write(get_seed_user_query())
            logger.info("Knowledge graph schema initialised.")
        except Exception as exc:
            logger.warning("Failed to initialise graph schema: %s", exc)
            client.close()
            return None

        # Build the extraction pipeline.
        use_spacy = getattr(settings, "graph_use_spacy", True)
        entity_extractor = RuleBasedEntityExtractor(use_spacy=use_spacy)
        relationship_extractor = RuleBasedRelationshipExtractor()

        service = GraphService(client=client)
        context_builder = GraphContextBuilder()

        return cls(
            client=client,
            service=service,
            entity_extractor=entity_extractor,
            relationship_extractor=relationship_extractor,
            context_builder=context_builder,
        )

    # -------------------------------------------------------------------
    # Extraction
    # -------------------------------------------------------------------

    def extract_and_store(self, text: str) -> None:
        """Extract entities and relationships from *text* and store in Neo4j.

        This is called after every user message, matching the pattern of
        ``MemoryManager.remember_from_user_message()``.
        """
        try:
            entities = self.entity_extractor.extract(text)
            if not entities:
                return

            relationships = self.relationship_extractor.extract(text, entities)

            for entity in entities:
                self.service.upsert_entity(entity)

            for relationship in relationships:
                self.service.add_relationship(relationship)

            logger.debug(
                "Graph extraction: %d entities, %d relationships from: %.60s...",
                len(entities),
                len(relationships),
                text,
            )
        except Exception as exc:
            # Never let graph extraction crash the main conversation loop.
            logger.warning("Graph extraction failed (non-fatal): %s", exc)

    # -------------------------------------------------------------------
    # Context building
    # -------------------------------------------------------------------

    def graph_context(self, user_input: str) -> str | None:
        """Build graph context relevant to the user's query.

        Strategy:
            1. Extract entities from the query.
            2. If specific entities are mentioned, fetch their neighbourhoods.
            3. Otherwise, return the user's general world model.
        """
        try:
            entities = self.entity_extractor.extract(user_input)

            # If the query mentions specific entities, build focused context.
            if entities:
                return self._entity_focused_context(entities)

            # Fall back to the user's general world model for broad queries
            # like "what do you know about me?" or "what am I working on?"
            return self._user_world_context()

        except Exception as exc:
            logger.warning("Graph context retrieval failed (non-fatal): %s", exc)
            return None

    def _entity_focused_context(self, entities: list[Entity]) -> str | None:
        """Build context for queries that mention specific entities."""
        sections: list[str] = []
        for entity in entities[:5]:  # Limit to avoid overwhelming the LLM.
            relationships = self.service.get_relationships(entity.name)
            section = self.context_builder.build_context_from_relationships(
                entity.name, relationships,
            )
            if section:
                sections.append(section)

        if not sections:
            return self._user_world_context()

        return "\n\n".join(sections)

    def _user_world_context(self) -> str | None:
        """Build context from the user's full world model."""
        user_context = self.service.get_user_context()
        return self.context_builder.build_context_from_user_world(user_context)

    # -------------------------------------------------------------------
    # CLI helpers
    # -------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return connection status and graph statistics."""
        connected = self.client.is_connected()
        stats = self.service.get_graph_stats() if connected else {}
        return {
            "connected": connected,
            "uri": self.client.uri,
            **stats,
        }

    def list_entities(self) -> list[dict[str, Any]]:
        """Return all entities grouped by type."""
        result: list[dict[str, Any]] = []
        for entity_type in EntityType:
            from jarvis.graph_memory import graph_queries
            records = self.client.execute_read(
                graph_queries.get_all_by_label_query(entity_type.value),
            )
            for record in records:
                result.append({
                    "type": entity_type.value,
                    "name": record["name"],
                    "canonical_name": record["canonical_name"],
                })
        return result

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def close(self) -> None:
        """Close the Neo4j connection."""
        self.client.close()
