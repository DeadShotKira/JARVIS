# JARVIS v0.4

Phase 4 memory-enabled local assistant with a persistent Personal Knowledge Graph (Neo4j) alongside document retrieval (RAG) and SQLite episodic memory.

## Goal

```text
User
  -> Terminal
  -> Python
  -> Memory Manager (SQLite)
  -> Graph Memory Manager (Neo4j)
  -> Document Retriever (ChromaDB)
  -> Context Builder & Aggregator
  -> Ollama
  -> Local model
  -> Memory / Graph Updaters
```

No cloud APIs. No paid services. No API keys.

## Quick Start

Install Python, Git, VS Code, Ollama, and Docker. Then check `config.yaml`:

```yaml
active_model: gemma3:12b
provider: ollama
```

Start Neo4j and Ollama via Docker Compose, pull the configured model, and run Jarvis:

```powershell
docker compose up -d
ollama run gemma3:12b
python main.py
```

Expected startup:

```text
JARVIS ONLINE
Brain: gemma3:12b
RAG: enabled
Graph: enabled
```

## Knowledge Commands

Phase 3 adds local document ingestion and retrieval:

```powershell
/knowledge add C:\path\to\DBMS_Notes.pdf
/knowledge list
/knowledge metadata DBMS_Notes.pdf
/knowledge remove DBMS_Notes.pdf
/knowledge rebuild
```

ChromaDB is the primary vector backend. FAISS is available behind the same
`VectorStore` abstraction for later backend swaps.

## Graph Commands

Phase 4 adds a persistent personal knowledge graph using Neo4j to store structured connections between entities (people, projects, technologies, interests, etc.):

```powershell
/graph status
/graph entities
/graph relationships Atharva
/graph search Neo4j
/graph extract "I'm building Jarvis using Python and Neo4j"
```

## Memory Test

Run Jarvis and say:

```text
I am building Jarvis.
```

Exit, restart, then ask:

```text
What projects am I working on?
```

Jarvis receives the stored project memory from SQLite and the relationships from Neo4j before generating the answer.

## Project Structure

```text
jarvis/
  main.py
  brain/
  config/
  database/
  docs/
  graph_memory/     # Phase 4 Knowledge Graph Memory
  knowledge/
    uploads/
    processed/
    indexes/
  memory/
  models/
  prompts/
  rag/
  requirements/
  tests/
```

## Configuration

Model selection is configuration-driven. To change the active brain, edit:

```yaml
active_model: gemma3:12b
```

RAG and Graph configuration are also fully configuration-driven:

```yaml
rag:
  vector_backend: chromadb
  embedding_provider: hashing
  embedding_model: local-hashing-v1

graph:
  enabled: true
  uri: bolt://localhost:7687
  username: neo4j
  password: jarvis_local
  extraction_backend: rule_based
  use_spacy: true
```

The application logic degrades gracefully if Neo4j is offline or Python dependencies are missing, letting JARVIS boot with the remaining memory layers.

## Tests

```powershell
python -m unittest discover -s jarvis/tests
```

## Phase Boundaries

Phase 4 implements a personal knowledge graph. It intentionally does not implement voice, agents, home automation, or vision.

