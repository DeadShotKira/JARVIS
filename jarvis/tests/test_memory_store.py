from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from jarvis.memory.memory_store import MemoryStore
from jarvis.memory.memory_types import MemoryType


class MemoryStoreTests(TestCase):
    def test_memory_insertion_and_database_operations(self):
        store = MemoryStore(":memory:")

        memory = store.add_memory(
            MemoryType.INTEREST,
            "Atharva enjoys Formula 1.",
            source_text="I love Formula 1.",
        )

        self.assertEqual(memory.memory_type, MemoryType.INTEREST)
        self.assertEqual(store.get_memory(memory.id).content, "Atharva enjoys Formula 1.")
        self.assertEqual(len(store.list_by_type(MemoryType.INTEREST)), 1)

    def test_memory_persistence_survives_reopen(self):
        with TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite3"
            first_store = MemoryStore(database_path)
            first_store.add_memory(
                MemoryType.PROJECT,
                "Atharva is building Jarvis.",
                source_text="I am building Jarvis.",
            )
            first_store.close()

            second_store = MemoryStore(database_path)
            memories = second_store.list_by_type(MemoryType.PROJECT)
            second_store.close()

        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].content, "Atharva is building Jarvis.")
