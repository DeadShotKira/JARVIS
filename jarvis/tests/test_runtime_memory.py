from unittest import TestCase

from jarvis.memory.runtime_memory import RuntimeMemory


class RuntimeMemoryTests(TestCase):
    def test_add_stores_messages(self):
        memory = RuntimeMemory(max_messages=4)

        memory.add("user", "Hello")
        memory.add("assistant", "Good evening, Atharva.")

        self.assertEqual(len(memory.messages), 2)
        self.assertEqual(memory.messages[0].role, "user")
        self.assertEqual(memory.messages[1].content, "Good evening, Atharva.")

    def test_memory_trims_oldest_messages(self):
        memory = RuntimeMemory(max_messages=2)

        memory.add("user", "one")
        memory.add("assistant", "two")
        memory.add("user", "three")

        self.assertEqual([message.content for message in memory.messages], ["two", "three"])
