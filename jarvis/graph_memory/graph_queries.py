"""Pre-built Cypher query templates for JARVIS Phase 4.

All Cypher lives here or in graph_service.py. Keeping query strings
separate from execution logic makes both easier to read and test.

Every query uses parameterised inputs ($name, $canonical_name, etc.)
to prevent injection and improve Neo4j query plan caching.

Design decisions:
    - MERGE is used everywhere instead of CREATE to ensure idempotency.
      Saying "I am learning Neo4j" three times should not create three nodes.
    - Label is injected as a string format (not a parameter) because Cypher
      does not support parameterised labels. This is safe because labels
      come from the EntityType enum, not user input.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Entity CRUD
# ---------------------------------------------------------------------------

def upsert_entity_query(label: str) -> str:
    """Return a MERGE query that creates or updates an entity node.

    The *label* is a Neo4j node label from EntityType (e.g. 'Technology').
    """
    return f"""
    MERGE (n:{label} {{canonical_name: $canonical_name}})
    ON CREATE SET
        n.name = $name,
        n.created_at = datetime(),
        n.updated_at = datetime(),
        n.source = $source
    ON MATCH SET
        n.updated_at = datetime()
    SET n += $properties
    RETURN n
    """


def get_entity_query(label: str) -> str:
    """Return a query to fetch a single entity by canonical_name."""
    return f"""
    MATCH (n:{label} {{canonical_name: $canonical_name}})
    RETURN n
    """


def delete_entity_query(label: str) -> str:
    """Return a query that detaches and deletes an entity."""
    return f"""
    MATCH (n:{label} {{canonical_name: $canonical_name}})
    DETACH DELETE n
    RETURN count(n) AS deleted
    """


def search_entities_query() -> str:
    """Full-text search across all entity types."""
    return """
    CALL db.index.fulltext.queryNodes("entity_search", $query)
    YIELD node, score
    RETURN labels(node) AS labels, node.name AS name,
           node.canonical_name AS canonical_name, score
    ORDER BY score DESC
    LIMIT $limit
    """


# ---------------------------------------------------------------------------
# Relationship CRUD
# ---------------------------------------------------------------------------

def upsert_relationship_query(
    source_label: str,
    target_label: str,
    rel_type: str,
) -> str:
    """Return a MERGE query that creates a relationship between two nodes.

    Both endpoint nodes are also MERGEd so the relationship write never
    fails due to a missing node.
    """
    return f"""
    MERGE (source:{source_label} {{canonical_name: $source_canonical}})
    ON CREATE SET
        source.name = $source_name,
        source.created_at = datetime(),
        source.source = $source_source
    MERGE (target:{target_label} {{canonical_name: $target_canonical}})
    ON CREATE SET
        target.name = $target_name,
        target.created_at = datetime(),
        target.source = $target_source
    MERGE (source)-[r:{rel_type}]->(target)
    ON CREATE SET r.created_at = datetime()
    SET r += $rel_properties
    RETURN source, r, target
    """


def get_relationships_query() -> str:
    """Get all relationships for a given entity (by canonical_name)."""
    return """
    MATCH (n {canonical_name: $canonical_name})-[r]-(neighbor)
    RETURN
        n.name AS entity,
        type(r) AS relationship,
        labels(neighbor) AS neighbor_labels,
        neighbor.name AS neighbor_name,
        neighbor.canonical_name AS neighbor_canonical,
        properties(r) AS rel_properties,
        startNode(r) = n AS outgoing
    ORDER BY relationship, neighbor_name
    """


def find_connections_query() -> str:
    """Find the shortest path between two entities (up to 4 hops)."""
    return """
    MATCH (a {canonical_name: $name_a}), (b {canonical_name: $name_b})
    MATCH path = shortestPath((a)-[*..4]-(b))
    RETURN
        [node IN nodes(path) | node.name] AS node_names,
        [rel IN relationships(path) | type(rel)] AS rel_types,
        length(path) AS hops
    """


# ---------------------------------------------------------------------------
# Convenience queries
# ---------------------------------------------------------------------------

def get_all_by_label_query(label: str) -> str:
    """Return all nodes of a given label."""
    return f"""
    MATCH (n:{label})
    RETURN n.name AS name, n.canonical_name AS canonical_name,
           properties(n) AS props
    ORDER BY n.name
    """


def get_projects_with_tech_query() -> str:
    """Return all projects with their associated technologies."""
    return """
    MATCH (p:Project)
    OPTIONAL MATCH (p)-[:USES]->(t:Technology)
    RETURN p.name AS project, p.canonical_name AS canonical_name,
           collect(t.name) AS technologies
    ORDER BY p.name
    """


def get_user_context_query() -> str:
    """Return everything directly connected to the user node.

    This is the 'world model' summary: projects, technologies, interests,
    organizations, and tasks linked to Atharva.
    """
    return """
    MATCH (user:Person {canonical_name: $user_canonical})-[r]-(connected)
    RETURN
        type(r) AS relationship,
        labels(connected) AS labels,
        connected.name AS name,
        properties(r) AS rel_properties
    ORDER BY relationship, name
    """


def get_entity_neighborhood_query() -> str:
    """Return an entity and its immediate neighbours (1-hop)."""
    return """
    MATCH (n {canonical_name: $canonical_name})-[r]-(neighbor)
    RETURN
        labels(n) AS entity_labels,
        n.name AS entity_name,
        type(r) AS relationship,
        labels(neighbor) AS neighbor_labels,
        neighbor.name AS neighbor_name
    ORDER BY relationship, neighbor_name
    """


def get_graph_stats_query() -> str:
    """Return aggregate counts of nodes and relationships."""
    return """
    CALL {
        MATCH (n) RETURN count(n) AS node_count
    }
    CALL {
        MATCH ()-[r]->() RETURN count(r) AS relationship_count
    }
    RETURN node_count, relationship_count
    """


def get_related_entities_query() -> str:
    """Return entities up to *depth* hops from a starting entity."""
    return """
    MATCH path = (start {canonical_name: $canonical_name})-[*1..2]-(related)
    WHERE start <> related
    RETURN DISTINCT
        labels(related) AS labels,
        related.name AS name,
        related.canonical_name AS canonical_name,
        length(path) AS distance
    ORDER BY distance, name
    LIMIT $limit
    """
