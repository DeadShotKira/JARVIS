from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from jarvis.rag.chunker import TextChunker
from jarvis.rag.context_builder import RagContextBuilder
from jarvis.rag.document_loader import DocumentLoader
from jarvis.rag.document_types import LoadedDocument
from jarvis.rag.embedder import HashingEmbedder
from jarvis.rag.rag_manager import RagManager
from jarvis.rag.retriever import DocumentRetriever
from jarvis.rag.vector_store import InMemoryVectorStore


class RagPipelineTests(TestCase):
    def test_loads_txt_document_with_metadata(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "notes.txt"
            path.write_text("Third Normal Form removes transitive dependencies.", encoding="utf-8")

            document = DocumentLoader().load(path)

        self.assertIn("Third Normal Form", document.text)
        self.assertEqual(document.metadata["filename"], "notes.txt")
        self.assertEqual(document.metadata["document_type"], "txt")

    def test_chunker_creates_overlapping_metadata_chunks(self):
        document = LoadedDocument(
            source_id="doc-1",
            text="Alpha beta gamma.\n\nDelta epsilon zeta.\n\nEta theta iota.",
            metadata={"filename": "notes.md", "document_type": "md"},
        )

        chunks = TextChunker(chunk_size=35, chunk_overlap=10).chunk(document)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].metadata["source_id"], "doc-1")
        self.assertEqual(chunks[0].metadata["chunk_id"], 0)

    def test_retriever_finds_semantically_related_chunk(self):
        embedder = HashingEmbedder(model_name="test-hashing", dimensions=64)
        store = InMemoryVectorStore()
        document = LoadedDocument(
            source_id="dbms",
            text="Third Normal Form removes transitive dependencies in database tables.",
            metadata={"filename": "DBMS_Notes.md", "document_type": "md"},
        )
        chunks = TextChunker(chunk_size=200, chunk_overlap=20).chunk(document)
        store.add_documents(chunks, embedder.embed_texts([chunk.text for chunk in chunks]))
        retriever = DocumentRetriever(
            embedder=embedder,
            vector_store=store,
            top_k=2,
            similarity_threshold=0.0,
        )

        results = retriever.retrieve("Explain transitive dependencies")

        self.assertEqual(results[0].chunk.metadata["filename"], "DBMS_Notes.md")
        self.assertGreater(results[0].score, 0)

    def test_context_builder_includes_memory_documents_and_question(self):
        embedder = HashingEmbedder(model_name="test-hashing", dimensions=64)
        store = InMemoryVectorStore()
        document = LoadedDocument(
            source_id="jarvis",
            text="Jarvis answers with source-aware document context.",
            metadata={"filename": "Jarvis.md", "document_type": "md"},
        )
        chunks = TextChunker(chunk_size=200, chunk_overlap=20).chunk(document)
        store.add_documents(chunks, embedder.embed_texts([chunk.text for chunk in chunks]))
        results = DocumentRetriever(embedder, store, top_k=1, similarity_threshold=0.0).retrieve(
            "How does Jarvis answer?"
        )

        context = RagContextBuilder().build(
            "How does Jarvis answer?",
            memory_context="Relevant long-term memories:\n- [PROJECT] Atharva is building Jarvis.",
            document_results=results,
        )

        self.assertIsNotNone(context)
        assert context is not None
        self.assertIn("Relevant long-term memories", context)
        self.assertIn("Relevant documents", context)
        self.assertIn("Jarvis.md", context)
        self.assertIn("User question", context)

    def test_rag_manager_ingests_lists_and_removes_document(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "UrbanEaze_Documentation.md"
            source.write_text(
                "The admin dashboard manages users, listings, analytics, and approvals.",
                encoding="utf-8",
            )
            embedder = HashingEmbedder(model_name="test-hashing", dimensions=64)
            store = InMemoryVectorStore()
            manager = RagManager(
                loader=DocumentLoader(),
                chunker=TextChunker(chunk_size=200, chunk_overlap=20),
                embedder=embedder,
                vector_store=store,
                retriever=DocumentRetriever(embedder, store, top_k=2, similarity_threshold=0.0),
                context_builder=RagContextBuilder(),
                uploads_path=root / "uploads",
                processed_path=root / "processed",
            )

            manifest = manager.ingest_document(source)
            documents = manager.list_documents()
            context = manager.build_context("What does the admin dashboard manage?")
            removed = manager.remove_document(manifest["source_id"])

        self.assertEqual(documents[0]["filename"], "UrbanEaze_Documentation.md")
        self.assertIsNotNone(context)
        assert context is not None
        self.assertIn("admin dashboard", context)
        self.assertTrue(removed)
