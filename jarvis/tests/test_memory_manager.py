from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from jarvis.memory.memory_manager import MemoryManager
from jarvis.memory.memory_utils import extract_memory_candidates


class MemoryManagerTests(TestCase):
    def test_extracts_project_memory_and_retrieves_after_restart(self):
        with TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "memory.sqlite3"
            first_manager = MemoryManager.from_database_path(str(database_path), retrieval_limit=5)
            first_manager.remember_from_user_message("I am building Jarvis.")
            first_manager.close()

            second_manager = MemoryManager.from_database_path(str(database_path), retrieval_limit=5)
            context = second_manager.recall_context("What projects am I working on?")
            second_manager.close()

        self.assertIsNotNone(context)
        self.assertIn("Atharva is building Jarvis.", context)

    def test_ignores_greetings_and_small_talk(self):
        self.assertEqual(extract_memory_candidates("Hello"), [])

    def test_extracts_supported_memory_categories(self):
        examples = {
            "I like dark themes.": "PREFERENCE",
            "I enjoy Formula 1.": "INTEREST",
            "Remind me to finish Phase 3.": "TASK",
            "My laptop has an RTX 3050.": "FACT",
        }

        for text, expected_type in examples.items():
            with self.subTest(text=text):
                candidates = extract_memory_candidates(text)
                self.assertEqual(candidates[0].memory_type.value, expected_type)
