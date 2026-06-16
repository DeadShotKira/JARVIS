# Phase 3 - Knowledge Base and RAG

JARVIS v0.3 adds a local Retrieval-Augmented Generation system. The goal is
simple: let Jarvis answer from user documents without hardcoding facts into the
model or sending data to cloud APIs.

## Why Embeddings Exist

Keyword search matches exact words. That is useful, but it misses meaning. A
question like "What does the admin panel control?" may need a document section
that says "dashboard responsibilities include approvals and analytics."

Embeddings solve this by turning text into vectors: lists of numbers that place
similar meanings near each other.

```text
"Third Normal Form removes transitive dependencies"
        |
        v
[0.12, -0.04, 0.31, ...]
```

Internally, retrieval compares the question vector with document chunk vectors.
Cosine similarity asks whether two vectors point in a similar direction. Higher
similarity means the text is more likely to be relevant.

Industry use: search, recommendation systems, support bots, research assistants,
and enterprise knowledge bases.

Jarvis use: retrieve local notes, PDFs, project docs, and personal knowledge
before calling the configured local model.

Raspberry Pi note: transformer embeddings may be slower on small hardware. The
embedding layer is configurable so Jarvis can use a tiny local model, a faster
hashing embedder for development, or another local embedding backend later.

## Why Chunking Exists

Models and vector databases work better with focused passages than full books.
Chunking splits documents into smaller overlapping sections.

```text
Document
  -> chunk 0: pages 1-2
  -> chunk 1: pages 2-3
  -> chunk 2: pages 3-4
```

Overlap prevents useful context from being cut in half at chunk boundaries.

Tradeoffs:

- Large chunks preserve context but can retrieve irrelevant text.
- Small chunks are precise but may lose surrounding explanation.
- Overlap improves recall but stores more text and vectors.

Jarvis config:

```yaml
rag:
  chunk_size: 900
  chunk_overlap: 150
```

## Vector Databases

SQLite is excellent for structured rows: memories, timestamps, metadata, and
settings. A vector database is built for nearest-neighbor search over embeddings.

```text
Question
  -> embedding
  -> vector database
  -> nearest document chunks
```

Phase 3 uses ChromaDB as the primary vector database. It is easier to inspect,
debug, reset, and learn with than raw FAISS indexes.

FAISS remains behind the `VectorStore` interface as a secondary adapter. Retrieval
code calls the abstraction, not ChromaDB directly, so the backend can change
without rewriting the assistant.

## RAG Flow

```text
User question
  -> memory retrieval
  -> document retrieval
  -> context builder
  -> configured local model
  -> source-aware answer
```

The context builder creates one package:

```text
Relevant long-term memories:
- ...

Relevant documents:
[1] Source: DBMS_Notes.pdf | Chunk: 3 | Similarity: 0.812
...

User question:
Explain Third Normal Form.
```

Best practices:

- Keep model names and paths in config.
- Preserve source metadata for citations.
- Keep document loading, chunking, embedding, storage, and retrieval separate.
- Tune `top_k` and `similarity_threshold` after testing with real documents.
- Rebuild indexes when changing embedding models.

Common mistakes:

- Hardcoding one vector backend into retrieval logic.
- Storing documents without filenames or source IDs.
- Using chunks that are too large for the model context window.
- Forgetting that PDF extraction quality varies by file.
