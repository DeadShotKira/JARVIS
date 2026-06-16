"""High-level assistant orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.brain.model_client import ModelClient
from jarvis.brain.ollama_client import ChatMessage
from jarvis.memory.memory_manager import MemoryManager
from jarvis.memory.runtime_memory import RuntimeMemory
from jarvis.rag.rag_manager import RagManager


@dataclass
class JarvisAssistant:
    """Coordinates prompts, runtime context, persistent memory, and model calls."""

    client: ModelClient
    runtime_memory: RuntimeMemory
    memory_manager: MemoryManager
    system_prompt: str
    rag_manager: RagManager | None = None

    def respond(self, user_input: str) -> str:
        """Send a user message to the local model and update memory."""
        messages = [ChatMessage(role="system", content=self.system_prompt)]

        memory_context = self.memory_manager.recall_context(user_input)
        retrieval_context = memory_context
        if self.rag_manager:
            retrieval_context = self.rag_manager.build_context(
                user_query=user_input,
                memory_context=memory_context,
            )

        if retrieval_context:
            messages.append(ChatMessage(role="system", content=retrieval_context))

        messages.extend(self.runtime_memory.messages)
        messages.append(ChatMessage(role="user", content=user_input))

        response = self.client.chat(messages)

        self.runtime_memory.add("user", user_input)
        self.runtime_memory.add("assistant", response)
        self.memory_manager.remember_from_user_message(user_input)
        return response
