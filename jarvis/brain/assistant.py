"""High-level assistant orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.brain.model_client import ModelClient
from jarvis.brain.ollama_client import ChatMessage
from jarvis.memory.memory_manager import MemoryManager
from jarvis.memory.runtime_memory import RuntimeMemory
from jarvis.rag.rag_manager import RagManager
from jarvis.graph_memory.graph_memory_manager import GraphMemoryManager



@dataclass
class JarvisAssistant:
    """Coordinates prompts, runtime context, persistent memory, and model calls."""

    client: ModelClient
    runtime_memory: RuntimeMemory
    memory_manager: MemoryManager
    system_prompt: str
    rag_manager: RagManager | None = None
    graph_manager: GraphMemoryManager | None = None

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

        graph_context = None
        if self.graph_manager:
            graph_context = self.graph_manager.graph_context(user_input)

        # Aggregate context from SQLite/RAG and Neo4j Graph
        context_parts = []
        if retrieval_context:
            context_parts.append(retrieval_context)
        if graph_context:
            context_parts.append(graph_context)

        if context_parts:
            messages.append(ChatMessage(role="system", content="\n\n".join(context_parts)))

        messages.extend(self.runtime_memory.messages)
        messages.append(ChatMessage(role="user", content=user_input))

        response = self.client.chat(messages)

        self.runtime_memory.add("user", user_input)
        self.runtime_memory.add("assistant", response)
        self.memory_manager.remember_from_user_message(user_input)

        if self.graph_manager:
            self.graph_manager.extract_and_store(user_input)

        return response

