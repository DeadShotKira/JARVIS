# JARVIS Knowledge Graph Memory (Phase 4)

This module implements a persistent **Personal Knowledge Graph** for JARVIS using **Neo4j**. It allows JARVIS to remember structured entities (people, projects, technologies, organizations, interests, tasks) and their connections, and injects this context directly into conversations.

## Architecture & Data Flow

The Knowledge Graph Memory layer works in parallel with SQLite episodic memory and ChromaDB vector search (RAG):

```
User Query
    ↓
┌──────────────────────────────────────┐
│         Context Gathering            │
│                                      │
│  ├── SQLite Memory (recall_context)  │  ← Preferences, facts, tasks
│  ├── ChromaDB RAG  (build_context)   │  ← Document chunks
│  └── Neo4j Graph   (graph_context)   │  ← Relationships & connections
│                                      │
│         Context Aggregator           │  ← Merges all context
│              ↓                       │
│         System Prompt                │
│              ↓                       │
│         OllamaClient.chat()          │
│              ↓                       │
│         Post-Processing              │
│  ├── RuntimeMemory.add()             │
│  ├── MemoryManager.remember()        │  ← SQLite extraction
│  └── GraphMemoryManager.extract()    │  ← Neo4j extraction (runs here)
└──────────────────────────────────────┘
```

## Schema Design

### Node Labels & Properties
- `(:Person)`: People JARVIS knows about (e.g. `Atharva` is seeded as the primary user). Unique constraint on `canonical_name`.
- `(:Project)`: Software projects, assignments, etc. (e.g. `Jarvis`).
- `(:Technology)`: Languages, libraries, databases, frameworks (e.g. `Python`, `Neo4j`).
- `(:Organization)`: Companies, schools, teams (e.g. `College`).
- `(:Interest)`: Hobbies and areas of curiosity (e.g. `Formula 1`, `AI`).
- `(:Task)`: Actionable items and reminders (e.g. `Build Android App`).
- `(:Document)`: References to ingested files (e.g. `DBMS_Notes.pdf`).

### Relationships (Edges)
- `(:Person)-[:WORKS_ON]->(:Project)`
- `(:Person)-[:USES]->(:Technology)`
- `(:Person)-[:LEARNS]->(:Technology)`
- `(:Person)-[:INTERESTED_IN]->(:Interest)`
- `(:Person)-[:BELONGS_TO]->(:Organization)`
- `(:Person)-[:HAS_TASK]->(:Task)`
- `(:Person)-[:KNOWS]->(:Person)`
- `(:Project)-[:USES]->(:Technology)`
- `(:Project)-[:HAS_DOCUMENT]->(:Document)`

*Note:* All writes use parameterized `MERGE` statements to ensure operations are idempotent. Unique constraints exist on each label's `canonical_name` (lowercase, normalized) to prevent duplicates while retaining display capitalization.

## Extraction Strategy

Extraction is fully local and runs in two steps after every user message:
1. **Entity Extraction** (`RuleBasedEntityExtractor`): Uses regex patterns for structured sentences (e.g., projects, interests, tasks, organizations), a case-insensitive lookup dictionary for common technologies, and **spaCy** Named Entity Recognition (`en_core_web_sm`) to discover person and organization names not covered by regex.
2. **Relationship Extraction** (`RuleBasedRelationshipExtractor`): Evaluates verb-phrase regex patterns over the text and links the extracted entities (e.g., "I'm learning Neo4j" links the user to `Neo4j` via `LEARNS`). It also infers implicit relationship edges if multiple matching nodes are found (e.g., user `USES` technology).

### Graceful Degradation
- **Neo4j down**: If the Neo4j container/instance is offline, JARVIS logs a warning and disables the graph module. Chat, SQLite, and RAG continue working normally.
- **spaCy missing**: If spaCy is not installed or `en_core_web_sm` is missing, the extractor falls back to pure regex-based extraction.

## Configuration (`config.yaml`)

```yaml
graph:
  enabled: true
  uri: bolt://localhost:7687
  username: neo4j
  password: jarvis_local
  database: neo4j
  extraction_backend: rule_based
  use_spacy: true
```

## Interactive CLI Commands

You can query and manage the graph memory directly in the JARVIS terminal loop:

- `/graph status` — Shows connection status, active URI, and total nodes/relationships.
- `/graph entities` — Lists all entities in the database grouped by their types.
- `/graph relationships <entity>` — Lists all 1-hop relationships (incoming and outgoing) for the specified entity name.
- `/graph search <query>` — Performs a full-text search against all entity names.
- `/graph extract <text>` — Runs the entity and relationship extraction pipeline dry-run on arbitrary text to debug matches.
