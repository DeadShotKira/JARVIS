# JARVIS v0.3

Phase 3 memory-enabled local assistant with a document knowledge base and RAG.

## Goal

```text
User
  -> Terminal
  -> Python
  -> Memory Manager
  -> Document Retriever
  -> Context Builder
  -> Ollama
  -> Local model
  -> Memory Updater
  -> SQLite
  -> ChromaDB
```

No cloud APIs. No paid services. No API keys.

## Quick Start

Install Python, Git, VS Code, and Ollama. Then check `config.yaml`:

```yaml
active_model: gemma3:4b
provider: ollama
```

Pull the configured model and start Jarvis:

```powershell
ollama run gemma3:4b
python main.py
```

Expected startup:

```text
JARVIS ONLINE
Brain: gemma3:4b
RAG: enabled
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

## Memory Test

Run Jarvis and say:

```text
I am building Jarvis.
```

Exit, restart, then ask:

```text
What projects am I working on?
```

Jarvis receives the stored project memory from SQLite before generating the answer.

## Project Structure

```text
jarvis/
  main.py
  brain/
  config/
  database/
  docs/
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
active_model: gemma3:4b
```

RAG is also configuration-driven:

```yaml
rag:
  vector_backend: chromadb
  embedding_provider: hashing
  embedding_model: local-hashing-v1
```

The application logic does not need to change when switching local models,
embedding providers, or vector backends.

## Tests

```powershell
python -m unittest discover -s jarvis/tests
```

## Phase Boundaries

Phase 3 implements local document RAG. It intentionally does not implement voice,
agents, home automation, or vision.
