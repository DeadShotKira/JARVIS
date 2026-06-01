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
