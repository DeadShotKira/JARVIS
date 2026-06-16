from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from jarvis.config.settings import load_settings


class SettingsTests(TestCase):
    def test_load_settings_uses_config_file(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "active_model: local-test-model",
                        "provider: ollama",
                        "ollama:",
                        "  host: http://localhost:11434",
                        "  request_timeout_seconds: 42",
                        "generation:",
                        "  temperature: 0.2",
                        "  context_window: 1024",
                        "memory:",
                        "  runtime_max_messages: 6",
                        "  database_path: data/test.sqlite3",
                        "  retrieval_limit: 3",
                        "rag:",
                        "  enabled: true",
                        "  vector_backend: memory",
                        "  collection_name: test_knowledge",
                        "  uploads_path: knowledge/uploads",
                        "  processed_path: knowledge/processed",
                        "  indexes_path: knowledge/indexes",
                        "  chunk_size: 500",
                        "  chunk_overlap: 50",
                        "  top_k: 2",
                        "  similarity_threshold: 0.1",
                        "  embedding_provider: hashing",
                        "  embedding_model: test-hashing",
                        "  embedding_dimensions: 64",
                        "prompts:",
                        "  personality_path: prompts/personality.md",
                    ]
                ),
                encoding="utf-8",
            )

            settings = load_settings(config_path)

        self.assertEqual(settings.active_model, "local-test-model")
        self.assertEqual(settings.runtime_max_messages, 6)
        self.assertEqual(settings.temperature, 0.2)
        self.assertEqual(settings.memory_database_path, (root / "data/test.sqlite3").resolve())
        self.assertTrue(settings.rag_enabled)
        self.assertEqual(settings.rag_vector_backend, "memory")
        self.assertEqual(settings.rag_chunk_size, 500)
        self.assertEqual(settings.rag_embedding_model, "test-hashing")
