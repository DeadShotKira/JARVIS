# JARVIS v0.2

Phase 2 memory-enabled local assistant foundation.

## Goal

```text
User
  -> Terminal
  -> Python
  -> Memory Manager
  -> Ollama
  -> Local model
  -> Memory Updater
  -> SQLite
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

Jarvis receives the stored project memory from SQLite before generating the answer.

## Project Structure

```text
jarvis/
├── main.py
├── brain/
├── config/
├── database/
├── docs/
├── memory/
├── models/
├── prompts/
├── requirements/
└── tests/
```

## Configuration

Model selection is configuration-driven. To change the active brain, edit:

```yaml
active_model: gemma3:4b
```

The application logic does not need to change when switching between local models.

## Tests

```powershell
python -m unittest discover -s jarvis/tests
```

## Phase Boundaries

Phase 2 implements persistent memory, categorization, retrieval, and SQLite storage. It intentionally does not implement RAG, voice, agents, home automation, or vision.
