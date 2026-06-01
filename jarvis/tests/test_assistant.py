from unittest import TestCase

from jarvis.brain.assistant import JarvisAssistant
from jarvis.brain.ollama_client import ChatMessage
from jarvis.memory.runtime_memory import RuntimeMemory


class FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, messages):
        self.calls.append(list(messages))
        return "Good evening, Atharva.\nHow may I assist?"


class FakeMemoryManager:
    def __init__(self):
        self.remembered = []

    def recall_context(self, user_input):
        if "project" in user_input.lower():
            return "Relevant long-term memories:\n- [PROJECT] Atharva is building Jarvis."
        return None

    def remember_from_user_message(self, user_input):
        self.remembered.append(user_input)
        return []


class AssistantTests(TestCase):
    def test_assistant_injects_system_prompt_and_updates_runtime_memory(self):
        client = FakeClient()
        runtime_memory = RuntimeMemory(max_messages=10)
        memory_manager = FakeMemoryManager()
        assistant = JarvisAssistant(
            client=client,
            runtime_memory=runtime_memory,
            memory_manager=memory_manager,
            system_prompt="You are Jarvis.",
        )

        response = assistant.respond("Hello")

        self.assertIn("Good evening", response)
        self.assertEqual(client.calls[0][0], ChatMessage(role="system", content="You are Jarvis."))
        self.assertEqual(runtime_memory.messages[0], ChatMessage(role="user", content="Hello"))
        self.assertEqual(runtime_memory.messages[1].role, "assistant")
        self.assertEqual(memory_manager.remembered, ["Hello"])

    def test_assistant_injects_relevant_persistent_memory(self):
        client = FakeClient()
        assistant = JarvisAssistant(
            client=client,
            runtime_memory=RuntimeMemory(max_messages=10),
            memory_manager=FakeMemoryManager(),
            system_prompt="You are Jarvis.",
        )

        assistant.respond("What projects am I working on?")

        self.assertEqual(client.calls[0][1].role, "system")
        self.assertIn("Atharva is building Jarvis.", client.calls[0][1].content)
