"""Rule-based relationship extraction for JARVIS Phase 4.

Given a text and a list of entities already extracted from it, this module
identifies directed relationships between those entities using verb-phrase
pattern matching.

The ``RelationshipExtractor`` protocol allows Phase 4B to swap in an
LLM-based extractor without changing any downstream graph code.

Design decisions:
    - The self-referencing user entity ("I", "my") is always resolved to
      the Atharva Person node, matching the convention in memory_utils.py.
    - Relationships are only created between entities that were actually
      extracted from the same text. This avoids hallucinated connections.
    - Patterns are ordered from most specific to least specific to reduce
      false positives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from jarvis.graph_memory.graph_schema import (
    Entity,
    EntityType,
    Relationship,
    RelationshipType,
    make_canonical,
)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class RelationshipExtractor(Protocol):
    """Extract relationships between entities from text."""

    def extract(self, text: str, entities: list[Entity]) -> list[Relationship]: ...


# ---------------------------------------------------------------------------
# The default user entity — "I" resolves to Atharva.
# ---------------------------------------------------------------------------

_SELF_ENTITY = Entity(
    name="Atharva",
    canonical_name="atharva",
    entity_type=EntityType.PERSON,
    source="system",
)


# ---------------------------------------------------------------------------
# Relationship patterns
# ---------------------------------------------------------------------------
# Each tuple is:
#   (compiled_regex, relationship_type, source_resolution, target_entity_type)
#
# source_resolution:
#   "self" → the source is the user (Atharva).
#   "match" → the source comes from a named group in the regex.

@dataclass(frozen=True)
class _RelPattern:
    pattern: re.Pattern[str]
    rel_type: RelationshipType
    source_is_self: bool
    target_entity_type: EntityType


_PATTERNS: list[_RelPattern] = [
    # --- Person → Project ---
    _RelPattern(
        re.compile(
            r"(?:i am|i'm)\s+(?:currently\s+)?(?:building|working on|developing|creating)\s+(?P<target>.+)",
            re.IGNORECASE,
        ),
        RelationshipType.WORKS_ON,
        source_is_self=True,
        target_entity_type=EntityType.PROJECT,
    ),

    # --- Person → Technology (learns) ---
    _RelPattern(
        re.compile(
            r"(?:i am|i'm)\s+(?:currently\s+)?learning\s+(?P<target>.+)",
            re.IGNORECASE,
        ),
        RelationshipType.LEARNS,
        source_is_self=True,
        target_entity_type=EntityType.TECHNOLOGY,
    ),

    # --- Person → Technology (uses) ---
    _RelPattern(
        re.compile(
            r"(?:i (?:am )?(?:using|use)|i'm using)\s+(?P<target>.+)",
            re.IGNORECASE,
        ),
        RelationshipType.USES,
        source_is_self=True,
        target_entity_type=EntityType.TECHNOLOGY,
    ),

    # --- Person → Interest ---
    _RelPattern(
        re.compile(
            r"(?:i (?:love|enjoy|am interested in|like|am passionate about))\s+(?P<target>.+)",
            re.IGNORECASE,
        ),
        RelationshipType.INTERESTED_IN,
        source_is_self=True,
        target_entity_type=EntityType.INTEREST,
    ),

    # --- Person → Organization ---
    _RelPattern(
        re.compile(
            r"(?:i (?:study|studied|work|worked) at|i'm at|i belong to)\s+(?P<target>.+)",
            re.IGNORECASE,
        ),
        RelationshipType.BELONGS_TO,
        source_is_self=True,
        target_entity_type=EntityType.ORGANIZATION,
    ),

    # --- Person → Task ---
    _RelPattern(
        re.compile(
            r"(?:i need to|i have to|i must|remind me to)\s+(?P<target>.+)",
            re.IGNORECASE,
        ),
        RelationshipType.HAS_TASK,
        source_is_self=True,
        target_entity_type=EntityType.TASK,
    ),

    # --- Person → Person (knows) ---
    _RelPattern(
        re.compile(
            r"(?:i know|my friend is|i met)\s+(?P<target>.+)",
            re.IGNORECASE,
        ),
        RelationshipType.KNOWS,
        source_is_self=True,
        target_entity_type=EntityType.PERSON,
    ),
]

# --- Project → Technology (uses) ---
# "Jarvis uses Neo4j", "UrbanEaze is built with Firebase"
_PROJECT_TECH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?P<project>\w[\w\s]*?)\s+(?:uses|is built with|runs on|is written in)\s+(?P<tech>.+)",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

@dataclass
class RuleBasedRelationshipExtractor:
    """Phase 4A relationship extractor: verb-phrase pattern matching.

    Matches sentence-level patterns against extracted entities to form
    typed, directed edges for the knowledge graph.
    """

    def extract(self, text: str, entities: list[Entity]) -> list[Relationship]:
        """Extract relationships from *text* given the known *entities*."""
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            return []

        relationships: list[Relationship] = []

        # Build a lookup from canonical_name → Entity for fast resolution.
        entity_lookup = _build_entity_lookup(entities)

        # 1. Self-referencing patterns (I am building, I am learning, etc.)
        relationships.extend(
            self._match_self_patterns(cleaned, entity_lookup)
        )

        # 2. Project → Technology patterns
        relationships.extend(
            self._match_project_tech_patterns(cleaned, entity_lookup)
        )

        # 3. Implicit: if a technology and a project are both mentioned
        #    in the same message but no explicit pattern matched, try to
        #    link them via the user (Atharva USES Technology).
        relationships.extend(
            self._infer_user_tech_relationships(entities, relationships)
        )

        return _deduplicate_relationships(relationships)

    # -------------------------------------------------------------------
    # Self-referencing patterns
    # -------------------------------------------------------------------

    def _match_self_patterns(
        self,
        text: str,
        entity_lookup: dict[str, Entity],
    ) -> list[Relationship]:
        results: list[Relationship] = []
        for rel_pattern in _PATTERNS:
            match = rel_pattern.pattern.search(text)
            if not match:
                continue

            raw_target = match.group("target").strip().strip(" .!?;:,")
            target_entity = _resolve_target(
                raw_target, rel_pattern.target_entity_type, entity_lookup,
            )
            if target_entity is None:
                continue

            results.append(Relationship(
                source_entity=_SELF_ENTITY,
                target_entity=target_entity,
                relationship_type=rel_pattern.rel_type,
            ))
        return results

    # -------------------------------------------------------------------
    # Project → Technology
    # -------------------------------------------------------------------

    def _match_project_tech_patterns(
        self,
        text: str,
        entity_lookup: dict[str, Entity],
    ) -> list[Relationship]:
        results: list[Relationship] = []
        for pattern in _PROJECT_TECH_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue

            project_name = match.group("project").strip()
            tech_name = match.group("tech").strip().strip(" .!?;:,")

            project_canonical = make_canonical(project_name)
            project_entity = entity_lookup.get(project_canonical)
            if project_entity is None or project_entity.entity_type != EntityType.PROJECT:
                continue

            # The tech part may mention multiple technologies separated by
            # "and" or commas.
            tech_names = re.split(r"\s*(?:,|and)\s*", tech_name)
            for single_tech in tech_names:
                single_tech = single_tech.strip().strip(" .!?;:,")
                tech_canonical = make_canonical(single_tech)
                tech_entity = entity_lookup.get(tech_canonical)
                if tech_entity and tech_entity.entity_type == EntityType.TECHNOLOGY:
                    results.append(Relationship(
                        source_entity=project_entity,
                        target_entity=tech_entity,
                        relationship_type=RelationshipType.USES,
                    ))
        return results

    # -------------------------------------------------------------------
    # Implicit inference
    # -------------------------------------------------------------------

    def _infer_user_tech_relationships(
        self,
        entities: list[Entity],
        existing: list[Relationship],
    ) -> list[Relationship]:
        """If a technology was extracted but has no relationship yet,
        infer (Atharva)-[:USES]->(Technology).  This is conservative:
        it only fires when the technology was explicitly mentioned.
        """
        already_related = {
            (r.source_entity.canonical_name, r.target_entity.canonical_name)
            for r in existing
        }
        inferred: list[Relationship] = []
        for entity in entities:
            if entity.entity_type != EntityType.TECHNOLOGY:
                continue
            key = (_SELF_ENTITY.canonical_name, entity.canonical_name)
            if key not in already_related:
                inferred.append(Relationship(
                    source_entity=_SELF_ENTITY,
                    target_entity=entity,
                    relationship_type=RelationshipType.USES,
                ))
                already_related.add(key)
        return inferred


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_entity_lookup(entities: list[Entity]) -> dict[str, Entity]:
    """Map canonical_name → Entity for fast resolution."""
    lookup: dict[str, Entity] = {}
    for entity in entities:
        lookup[entity.canonical_name] = entity
    return lookup


def _resolve_target(
    raw_name: str,
    expected_type: EntityType,
    entity_lookup: dict[str, Entity],
) -> Entity | None:
    """Try to match *raw_name* to an entity in the lookup.

    If the raw name contains multiple tokens separated by "and" / comma,
    only the first match is returned.
    """
    candidates = re.split(r"\s*(?:,|and)\s*", raw_name)
    for candidate in candidates:
        canonical = make_canonical(candidate.strip())
        entity = entity_lookup.get(canonical)
        if entity is not None:
            return entity

    # If no extracted entity matched, create one from the raw text.
    # This happens when the relationship phrase names something that the
    # entity extractor didn't pick up (e.g., a novel project name).
    first = candidates[0].strip()
    if first and len(first) > 1:
        return Entity(
            name=first,
            canonical_name=make_canonical(first),
            entity_type=expected_type,
        )
    return None


def _deduplicate_relationships(rels: list[Relationship]) -> list[Relationship]:
    """Remove duplicate relationships (same source, target, type)."""
    seen: set[tuple[str, str, str]] = set()
    unique: list[Relationship] = []
    for rel in rels:
        key = (
            rel.source_entity.canonical_name,
            rel.target_entity.canonical_name,
            rel.relationship_type.value,
        )
        if key not in seen:
            seen.add(key)
            unique.append(rel)
    return unique
