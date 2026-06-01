"""Command-line interface for JARVIS v0.2."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from jarvis.brain.assistant import JarvisAssistant
from jarvis.brain.ollama_client import OllamaConnectionError
from jarvis.brain.provider_factory import create_model_client
from jarvis.config.settings import Settings, load_settings
from jarvis.memory.memory_manager import MemoryManager
from jarvis.memory.runtime_memory import RuntimeMemory
from jarvis.prompts.loader import load_personality_prompt


def build_assistant(settings: Settings | None = None) -> JarvisAssistant:
    """Create a configured assistant instance."""
    resolved_settings = settings or load_settings()
    client = create_model_client(resolved_settings)
    runtime_memory = RuntimeMemory(max_messages=resolved_settings.runtime_max_messages)
    memory_manager = MemoryManager.from_database_path(
        str(resolved_settings.memory_database_path),
        retrieval_limit=resolved_settings.memory_retrieval_limit,
    )
    personality_prompt = load_personality_prompt(resolved_settings.personality_path)

    return JarvisAssistant(
        client=client,
        runtime_memory=runtime_memory,
        memory_manager=memory_manager,
        system_prompt=personality_prompt,
    )


def main() -> None:
    """Run the interactive terminal loop."""
    settings = load_settings()
    assistant = build_assistant(settings)

    print("JARVIS ONLINE")
    print(f"Brain: {settings.active_model}")
    print("Type 'exit' or 'quit' to shut down.\n")

    while True:
        try:
            user_input = input("You:\n").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nJarvis:\nSession terminated. Naturally, I had it under control.")
            return

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("\nJarvis:\nPowering down. Try not to miss me too much.")
            return

        try:
            response = assistant.respond(user_input)
        except OllamaConnectionError as exc:
            print("\nJarvis:")
            print(exc)
            return

        print("\nJarvis:")
        print(response)
        print()


if __name__ == "__main__":
    main()
