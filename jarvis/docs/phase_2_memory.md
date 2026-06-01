# Phase 2 Memory System

## 2.1 Context, Runtime Memory, and Persistent Memory

What problem this solves: A chatbot that only sees the current message cannot build continuity. Memory lets Jarvis carry useful information across time.

Why it exists: Human assistants improve because they remember preferences, projects, constraints, and goals. AI assistants need explicit systems for that.

Internal implementation:

```text
Conversation context: messages sent to the model right now
Runtime memory: recent messages kept in RAM while the app is running
Persistent memory: durable facts stored in SQLite across restarts
```

Example:

```text
User: I love Formula 1.
Memory stored: [INTEREST] Atharva enjoys Formula 1.

One week later:
User: What do you remember about my interests?
Jarvis context receives: [INTEREST] Atharva enjoys Formula 1.
```

Industry usage: Production assistants usually separate raw chat logs, durable user profile facts, task state, and semantic retrieval indexes.

How Jarvis uses it: Phase 2 stores selected facts in SQLite, retrieves relevant memories before each model call, and injects them into the prompt.

## 2.2 Database Fundamentals

A database stores structured information.

Table: A named collection of related records.

Row: One record in a table.

Column: One field inside each row.

Primary key: A stable ID that uniquely identifies a row.

Query: A request for data, usually written in SQL.

Jarvis uses a `memories` table:

```text
id | memory_type | content                       | importance
1  | PROJECT     | Atharva is building Jarvis.   | 4
2  | INTEREST    | Atharva enjoys Formula 1.     | 3
```

Why SQLite is chosen: It is local, file-based, reliable, included with Python, fast enough for small personal memory, and easy to run on Raspberry Pi.

Raspberry Pi implications: SQLite avoids running a separate database server, saves RAM, and keeps deployment simple.

## 2.3 Persistent Memory Storage

What problem it solves: Jarvis can remember after the Python process exits.

Internal implementation: `MemoryStore` opens SQLite, creates tables, inserts structured memories, and reads them later.

How Jarvis uses it: User statements like `I am building Jarvis.` become durable rows such as:

```text
[PROJECT] Atharva is building Jarvis.
```

## 2.4 Memory Categorization

What problem it solves: Different memories need different treatment.

Current categories:

- `PREFERENCE`: "Atharva likes dark themes."
- `INTEREST`: "Atharva enjoys Formula 1."
- `PROJECT`: "Atharva is building Jarvis."
- `TASK`: "Atharva wants to be reminded to finish Phase 3."
- `FACT`: "Atharva's laptop has an RTX 3050."

Internal implementation: `MemoryType` is an enum. This keeps the database consistent and avoids loose string labels spreading through the code.

Industry usage: Assistants commonly separate preferences, profile facts, tasks, project state, and conversation history.

## 2.5 Memory Retrieval

What problem it solves: The model cannot use a memory unless it appears in the current context.

Internal implementation: `MemoryRetriever` scores stored memories against the current user message using keyword overlap and memory-type hints.

Example:

```text
User: What projects am I working on?
Retrieved: [PROJECT] Atharva is building Jarvis.
```

How Jarvis uses it: Retrieved memories are inserted as a system context message before the model is called.

## 2.6 Memory Creation Logic

What problem it solves: Storing everything creates noise, privacy risk, and bad retrieval.

Jarvis stores:

- Preferences
- Long-term interests
- Projects
- Goals and tasks
- Important facts

Jarvis ignores:

- Greetings
- Casual small talk
- One-off temporary statements
- Most questions

Internal implementation: Phase 2 uses deterministic extraction rules in `memory_utils.py`. This is transparent, testable, and offline.

Common beginner mistake: Treating memory as a full transcript. Memory should be curated facts, not a junk drawer.

## 2.7 Embeddings Introduction

What problem embeddings solve: Exact keyword search misses meaning. "car racing" and "Formula 1" are related even when words differ.

Why vector search exists: Embeddings turn text into lists of numbers where similar meanings are closer together.

Internal implementation: An embedding model converts text into a vector. A vector database or vector index compares vectors using similarity math.

How semantic retrieval works:

```text
"I enjoy Formula 1." -> [0.12, -0.44, 0.08, ...]
"What sports do I follow?" -> [0.10, -0.39, 0.11, ...]
Similarity score says these are related.
```

Industry usage: RAG systems use embeddings to retrieve relevant documents, notes, and memories before generation.

How Jarvis prepares for it: The database already has a `memory_embeddings` table. Phase 2 does not fill it yet. Phase 3 can add local embeddings without redesigning the database.

Raspberry Pi considerations: Embedding models must be small and local. A lightweight embedding model is usually better than a giant model that makes the assistant slow.

## 2.8 Testing

Phase 2 tests cover:

- Memory insertion
- Memory retrieval
- Persistence across database reopen
- Database operations
- Config-driven model loading
- Assistant prompt injection

Run:

```powershell
python -m unittest discover -s jarvis/tests
```
