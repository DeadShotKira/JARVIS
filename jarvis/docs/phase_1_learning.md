# Phase 1 Learning Notes

## Phase 1.1: Local LLM Concepts

### Inference

What problem it solves: Inference turns your prompt into a model response.

Why it exists: Training creates the model. Inference uses the trained model.

How it works internally: Text is tokenized, passed through transformer layers, and the model predicts the next token repeatedly until the answer is complete.

How Jarvis will use it: Every terminal message goes to Ollama, which runs local inference and returns text.

Raspberry Pi considerations: Inference speed depends heavily on RAM bandwidth, CPU, model size, and quantization. Small models matter more than flashy benchmarks.

Common beginner mistakes: Expecting a small model to behave like a massive cloud model, ignoring token limits, and running models too large for available memory.

Industry best practice: Start with the smallest model that reliably solves the task, then scale only when there is a measured need.

### VRAM and RAM

What problem it solves: Memory stores model weights and the active context while the model runs.

Why it exists: Neural networks are large numeric matrices. Those numbers must live somewhere during inference.

How it works internally: GPU deployments use VRAM. CPU deployments use system RAM. Raspberry Pi runs mostly through CPU and shared system memory.

How Jarvis will use it: The Windows laptop may use GPU acceleration, but Phase 1 architecture must still run without a dedicated GPU.

Raspberry Pi considerations: A Raspberry Pi 5 with 8GB or 16GB RAM needs small quantized models. Leave memory for the OS and future services.

Common beginner mistakes: Choosing a model because it is popular rather than because it fits memory limits.

Industry best practice: Measure memory, latency, and output quality together.

### Quantization

What problem it solves: Quantization makes models smaller and faster.

Why it exists: Full precision weights are expensive to store and compute. Many assistant tasks work well with lower precision.

How it works internally: Model weights are represented with fewer bits, often 4-bit or 8-bit instead of 16-bit.

How Jarvis will use it: Ollama normally serves quantized model builds suitable for local devices.

Raspberry Pi considerations: Quantization is the difference between "runs nicely" and "why is my Pi suffering?"

Common beginner mistakes: Assuming more bits always means better real-world assistant behavior.

Industry best practice: Use quantized models for edge deployment and validate quality with task-specific prompts.

### GGUF

What problem it solves: GGUF packages model weights and metadata in a local inference-friendly format.

Why it exists: Local runtimes need a portable model format that supports quantization and metadata.

How it works internally: A GGUF file stores tensors, tokenizer information, and model metadata used by runtimes such as llama.cpp-based systems.

How Jarvis will use it: Ollama abstracts most GGUF details, but the underlying local model may be stored in a quantized format.

Raspberry Pi considerations: GGUF-style quantized models are central to small-device local AI.

Common beginner mistakes: Downloading random model files without knowing whether the local runtime supports the architecture.

Industry best practice: Use well-maintained model distributions and track model name, size, quantization, and license.

### Ollama Architecture

What problem it solves: Ollama provides a simple local server for downloading, storing, and running models.

Why it exists: Without a local model server, every app would need to manage model files, inference processes, and HTTP APIs itself.

How it works internally: Ollama runs a local service, stores models, exposes endpoints such as `/api/chat`, and starts model runners as needed.

How Jarvis will use it: Python sends chat messages to `http://localhost:11434/api/chat`.

Raspberry Pi considerations: Keep the Python app separate from the model server. That makes Docker and service management easier later.

Common beginner mistakes: Thinking `ollama run` is the only interface. It is also a local API server.

Industry best practice: Treat the model server as a replaceable local service behind a clean client module.

### Model Files

What problem it solves: Model files store the learned behavior of the AI.

Why it exists: The assistant code is tiny compared with the model weights.

How it works internally: Files contain weight tensors and tokenizer data. Ollama stores and manages them outside this repository.

How Jarvis will use it: This repo stores configuration and code, not huge model binaries.

Raspberry Pi considerations: Keep model files out of Git and choose models that fit local storage and memory.

Common beginner mistakes: Committing model files into the project repository.

Industry best practice: Document model tags and setup commands instead of storing binaries in source control.

## Phase 1.2: Environment Setup

Python runs the Jarvis application logic. Git tracks project changes. VS Code gives an editor and terminal. Ollama runs the local LLM.

They interact like this:

```text
Terminal -> Python main.py -> Ollama local HTTP API -> Local model -> Python -> Terminal
```

Install and verify:

```powershell
python --version
git --version
ollama --version
ollama run qwen3:1.7b
python main.py
```

## Phase 1.3: Model Evaluation

Phase 1 prioritizes efficiency over raw intelligence.

Recommended Phase 1 brain: `qwen3:1.7b`

Why: It is small enough for local-first development, more capable than ultra-tiny models, and a realistic Raspberry Pi candidate when quantized.

Fallback for slower Pi deployments: `tinyllama:1.1b`

## Phase 1.4: Prompt Engineering

System prompts define assistant identity and rules. User prompts are Atharva's messages. Context is the set of messages sent to the model. Conversation state is the short runtime history stored in RAM.

Jarvis uses `prompts/personality.md` as the injected system prompt.

## Phase 1.5: Python + Ollama Integration

The integration lives in `brain/ollama_client.py`. It sends JSON to Ollama's local `/api/chat` endpoint and returns the assistant response.

No cloud keys. No paid API. No vendor lock-in. Delightfully unfashionable, which is often how robust systems begin.

## Phase 1.6: Conversation Memory

Runtime memory stores recent messages in RAM only. It is reset when the program exits.

This avoids premature databases and keeps Phase 1 understandable.

## Phase 1.7: Personality Engine

The personality engine is intentionally simple: load Markdown, inject it as the system prompt, and keep it at the front of every model call.

## Phase 1.8: Project Architecture

The package is split into small modules:

```text
jarvis/
├── main.py
├── brain/
├── prompts/
├── memory/
├── config/
├── models/
├── docs/
├── tests/
└── requirements/
```

This keeps the future path open for memory, RAG, speech, Android, home automation, agents, and Raspberry Pi deployment.
