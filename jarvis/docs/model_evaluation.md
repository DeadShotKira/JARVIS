# Phase 1 Model Evaluation

These estimates are practical starting points. Exact RAM, speed, and quality depend on the model tag, quantization, Ollama version, context length, operating system, and what else is running.

| Family | Example Ollama tag | RAM/VRAM profile | Speed | Reasoning | Raspberry Pi suitability |
| --- | --- | --- | --- | --- | --- |
| Qwen | `qwen3:1.7b` | Low to moderate | Good | Good for size | Best Phase 1 default |
| Gemma | `gemma3:1b` | Low | Good | Good general chat | Strong candidate, verify exact tag and license |
| TinyLlama | `tinyllama:1.1b` | Very low | Very fast | Limited | Best fallback for weak hardware |
| Phi | `phi3:mini` | Moderate | Medium | Strong small-model reasoning | Useful, but can be heavier |

## Selection

Phase 1 Jarvis brain: `qwen3:1.7b`

Rationale:
- Efficiency matters more than maximum intelligence.
- It keeps development realistic for Raspberry Pi 5.
- It should be capable enough for terminal assistant behavior.
- It leaves architectural room for later RAG and tool use.

## Test Prompts

Use the same prompts for each model:

```text
Explain quantization to a beginner in 5 bullet points.
Remember that my project is called JARVIS. What is it called?
Give me a Raspberry Pi-safe architecture for a local assistant.
Respond like Jarvis, but keep it professional.
```

Track:
- First-token latency
- Total response time
- RAM usage
- CPU/GPU usage
- Response usefulness
- Whether it follows the personality prompt
