# JARVIS v0.2 Architecture

```text
User
  |
  v
Terminal
  |
  v
Python CLI
  |
  v
JarvisAssistant
  |
  +--> ProviderFactory
  |      |
  |      v
  |    ModelClient
  |
  +--> RuntimeMemory
  |
  +--> MemoryManager
  |      |
  |      +--> MemoryRetriever
  |      |
  |      +--> MemoryStore
  |              |
  |              v
  |            SQLite
  |
  v
Configured ModelClient
  |
  v
Ollama local API
  |
  v
Configured local model
```

## Request Flow

1. The user sends a terminal message.
2. `MemoryManager` retrieves relevant long-term memories from SQLite.
3. `JarvisAssistant` injects the personality prompt, memory context, runtime context, and user message.
4. `OllamaClient` sends the request to the configured local model.
5. Jarvis stores the current chat turn in runtime memory.
6. `MemoryManager` decides whether the user message contains durable information.
7. Durable memories are written to SQLite.

## Design Principles

- Local-first: no cloud calls or API keys.
- Modular: model, memory, prompts, and database are separate.
- Raspberry Pi compatible: SQLite, small context windows, and standard-library-first code.
- Docker friendly: model, database, prompts, and provider are loaded from config.
- Future-ready: embeddings have a database table, but Phase 2 does not implement RAG.

## Future Expansion Slots

- `memory/memory_retriever.py`: replace or augment lexical retrieval with embeddings.
- `database/`: add migrations when schema changes become more complex.
- `brain/`: add provider routing for future local or cloud models.
- `prompts/`: split system prompts by mode or task.
