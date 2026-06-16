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
from jarvis.rag.rag_manager import RagManager


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
    rag_manager = RagManager.from_settings(resolved_settings) if resolved_settings.rag_enabled else None

    return JarvisAssistant(
        client=client,
        runtime_memory=runtime_memory,
        memory_manager=memory_manager,
        system_prompt=personality_prompt,
        rag_manager=rag_manager,
    )


def main() -> None:
    """Run the interactive terminal loop."""
    settings = load_settings()
    assistant = build_assistant(settings)

    print("JARVIS ONLINE")
    print(f"Brain: {settings.active_model}")
    print(f"RAG: {'enabled' if settings.rag_enabled else 'disabled'}")
    print("Type 'exit' or 'quit' to shut down.\n")
    print("Knowledge commands: /knowledge add <path>, list, remove <id|filename>, rebuild, metadata <id|filename>\n")

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

        if user_input.startswith("/knowledge"):
            print("\nJarvis:")
            print(handle_knowledge_command(assistant, user_input))
            print()
            continue

        try:
            response = assistant.respond(user_input)
        except OllamaConnectionError as exc:
            print("\nJarvis:")
            print(exc)
            return

        print("\nJarvis:")
        print(response)
        print()


def handle_knowledge_command(assistant: JarvisAssistant, command: str) -> str:
    """Handle local knowledge-base commands."""
    if assistant.rag_manager is None:
        return "RAG is disabled in configuration."

    parts = command.strip().split(maxsplit=2)
    if len(parts) == 1:
        return _knowledge_help()

    action = parts[1].lower()
    argument = parts[2].strip().strip('"') if len(parts) > 2 else ""

    if action == "add":
        if not argument:
            return "Usage: /knowledge add <path>"
        manifest = assistant.rag_manager.ingest_document(argument)
        return (
            f"Indexed {manifest['filename']} "
            f"({manifest['chunk_count']} chunks, source_id={manifest['source_id']})."
        )

    if action == "list":
        documents = assistant.rag_manager.list_documents()
        if not documents:
            return "No documents indexed yet."
        return "\n".join(
            f"- {doc.get('filename')} | chunks={doc.get('chunk_count')} | "
            f"source_id={doc.get('source_id')}"
            for doc in documents
        )

    if action == "remove":
        if not argument:
            return "Usage: /knowledge remove <source_id|filename>"
        removed = assistant.rag_manager.remove_document(argument)
        return "Document removed." if removed else "Document not found."

    if action == "rebuild":
        manifests = assistant.rag_manager.rebuild_index()
        return f"Rebuilt index with {len(manifests)} document(s)."

    if action == "metadata":
        if not argument:
            return "Usage: /knowledge metadata <source_id|filename>"
        metadata = assistant.rag_manager.document_metadata(argument)
        if not metadata:
            return "Document not found."
        return "\n".join(f"{key}: {value}" for key, value in sorted(metadata.items()))

    return _knowledge_help()


def _knowledge_help() -> str:
    return (
        "Usage: /knowledge add <path> | /knowledge list | "
        "/knowledge remove <source_id|filename> | /knowledge rebuild | "
        "/knowledge metadata <source_id|filename>"
    )


if __name__ == "__main__":
    main()
