from pathlib import Path
from unittest import TestCase

from jarvis.brain.ollama_client import OllamaClient
from jarvis.brain.provider_factory import create_model_client
from jarvis.config.settings import Settings


class ProviderFactoryTests(TestCase):
    def test_creates_ollama_client_from_configured_model(self):
        settings = Settings(
            active_model="local-test-model",
            provider="ollama",
            ollama_host="http://localhost:11434",
            request_timeout_seconds=30,
            runtime_max_messages=4,
            memory_database_path=Path("memory.sqlite3"),
            memory_retrieval_limit=3,
            temperature=0.1,
            context_window=512,
            personality_path=Path("personality.md"),
            rag_enabled=False,
            rag_vector_backend="memory",
            rag_collection_name="test",
            rag_uploads_path=Path("uploads"),
            rag_processed_path=Path("processed"),
            rag_indexes_path=Path("indexes"),
            rag_chunk_size=500,
            rag_chunk_overlap=50,
            rag_top_k=2,
            rag_similarity_threshold=0.1,
            rag_embedding_provider="hashing",
            rag_embedding_model="test-hashing",
            rag_embedding_dimensions=64,
        )

        client = create_model_client(settings)

        self.assertIsInstance(client, OllamaClient)
        self.assertEqual(client.model, "local-test-model")
