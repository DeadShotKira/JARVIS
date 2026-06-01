# Phase 2 Setup

## Tools

Python runs the Jarvis application.

Git tracks project changes.

VS Code gives an editor and terminal.

Ollama runs the configured local model.

SQLite stores long-term memory in a local file.

## Windows Development Setup

```powershell
python --version
git --version
ollama --version
ollama run gemma3:4b
python main.py
```

## Configuration

Jarvis reads `config.yaml` by default:

```yaml
active_model: gemma3:4b
provider: ollama

memory:
  database_path: data/jarvis_memory.sqlite3
```

To use a different config file:

```powershell
$env:JARVIS_CONFIG_PATH = "C:\path\to\config.yaml"
python main.py
```

## Raspberry Pi Direction

The code avoids GPU-only assumptions and cloud APIs. On Raspberry Pi, install Python and Ollama, pick a small quantized model in `config.yaml`, and run the same Python entry point.

SQLite is a good Phase 2 fit because it is local, reliable, built into Python, and light enough for Raspberry Pi storage.
