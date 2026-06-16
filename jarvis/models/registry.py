"""Phase 1 model candidates.

These are recommendations, not downloaded model files. Ollama manages model
storage outside this repository.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCandidate:
    name: str
    ollama_tag: str
    strengths: str
    pi_suitability: str


PHASE_1_CANDIDATES = [
    ModelCandidate(
        name="Qwen",
        ollama_tag="qwen3:1.7b",
        strengths="Best Phase 1 balance of reasoning, size, and instruction following.",
        pi_suitability="Good on Raspberry Pi 5 with enough RAM, especially quantized.",
    ),
    ModelCandidate(
        name="Gemma",
        ollama_tag="gemma3:1b",
        strengths="Efficient Google-family model with good general chat behavior.",
        pi_suitability="Good, but test licensing and exact tag availability before deployment.",
    ),
    ModelCandidate(
    name="Gemma",
    ollama_tag="gemma3:12b",
    strengths="Strong general-purpose assistant with excellent instruction following, reasoning, and conversational quality.",
    pi_suitability="Not intended for Raspberry Pi deployment. Best suited for desktop-class systems with at least 16 GB RAM.",
    ),
    ModelCandidate(
        name="Gemma 4",
        ollama_tag="gemma4:26b",
        strengths="Much stronger reasoning, coding, and instruction following than the small Gemma models.",
        pi_suitability="Not suitable for Raspberry Pi deployment. Intended for powerful desktops, servers, or cloud GPUs.",
    ),
    ModelCandidate(
        name="TinyLlama",
        ollama_tag="tinyllama:1.1b",
        strengths="Very light and fast for simple tasks.",
        pi_suitability="Excellent speed, weaker reasoning.",
    ),
    ModelCandidate(
        name="Phi",
        ollama_tag="phi3:mini",
        strengths="Strong small-model reasoning.",
        pi_suitability="Capable, but mini variants can be heavier than ideal for Pi-first work.",
    ),
]
