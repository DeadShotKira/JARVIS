import sys
import unittest
from unittest.mock import MagicMock, patch


# Mock neo4j library for environment compatibility if it is not installed
try:
    import neo4j
except ImportError:
    mock_neo4j = MagicMock()
    sys.modules['neo4j'] = mock_neo4j

from jarvis.graph_memory.graph_schema import (
    Entity,
    EntityType,
    Relationship,
    RelationshipType,
    make_canonical,
)
from jarvis.graph_memory.entity_extractor import RuleBasedEntityExtractor
from jarvis.graph_memory.relationship_extractor import (
    RuleBasedRelationshipExtractor,
    _SELF_ENTITY,
)
from jarvis.graph_memory.graph_context_builder import GraphContextBuilder
from jarvis.graph_memory.neo4j_client import Neo4jClient, Neo4jConnectionError
from jarvis.graph_memory.graph_service import GraphService
from jarvis.graph_memory.graph_memory_manager import GraphMemoryManager


class TestEntityExtractor(unittest.TestCase):
    def setUp(self):
        # Disable spaCy for deterministic regex unit tests unless explicitly testing it
        self.extractor = RuleBasedEntityExtractor(use_spacy=False)

    def test_extract_technologies(self):
        text = "I am using Python, Neo4j, and Docker."
        entities = self.extractor.extract(text)
        tech_names = {e.name for e in entities if e.entity_type == EntityType.TECHNOLOGY}
        self.assertIn("Python", tech_names)
        self.assertIn("Neo4j", tech_names)
        self.assertIn("Docker", tech_names)

        # Single letter boundary checks
        text_c = "I write in C and R."
        entities_c = self.extractor.extract(text_c)
        tech_c = {e.name for e in entities_c if e.entity_type == EntityType.TECHNOLOGY}
        self.assertIn("C", tech_c)
        self.assertIn("R", tech_c)


    def test_extract_projects(self):
        texts = [
            "I'm currently building Jarvis",
            "my project is called UrbanEaze",
            "This is the website project",
        ]
        expected_names = ["Jarvis", "UrbanEaze", "website"]

        for text, expected in zip(texts, expected_names):
            entities = self.extractor.extract(text)
            project_entities = [e for e in entities if e.entity_type == EntityType.PROJECT]
            self.assertTrue(len(project_entities) >= 1)
            self.assertEqual(project_entities[0].name, expected)

    def test_extract_interests(self):
        texts = [
            "I am interested in Formula 1 racing",
            "my hobbies include coding and photography",
            "I like reading science fiction",
        ]
        expected_names = ["Formula 1 racing", "coding and photography", "reading science fiction"]

        for text, expected in zip(texts, expected_names):
            entities = self.extractor.extract(text)
            interest_entities = [e for e in entities if e.entity_type == EntityType.INTEREST]
            self.assertTrue(len(interest_entities) >= 1)
            self.assertEqual(interest_entities[0].name, expected)

    def test_extract_organizations(self):
        text = "I study at Stanford University."
        entities = self.extractor.extract(text)
        org_entities = [e for e in entities if e.entity_type == EntityType.ORGANIZATION]
        self.assertEqual(len(org_entities), 1)
        self.assertEqual(org_entities[0].name, "Stanford University")

    def test_extract_tasks(self):
        text = "remind me to build the user interface"
        entities = self.extractor.extract(text)
        task_entities = [e for e in entities if e.entity_type == EntityType.TASK]
        self.assertEqual(len(task_entities), 1)
        self.assertEqual(task_entities[0].name, "build the user interface")


class TestRelationshipExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = RuleBasedRelationshipExtractor()

    def test_extract_self_relationships(self):
        # 1. WORKS_ON
        entities = [Entity("Jarvis", "jarvis", EntityType.PROJECT)]
        rels = self.extractor.extract("I'm building Jarvis", entities)
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0].source_entity, _SELF_ENTITY)
        self.assertEqual(rels[0].target_entity.name, "Jarvis")
        self.assertEqual(rels[0].relationship_type, RelationshipType.WORKS_ON)

        # 2. LEARNS
        entities_learn = [Entity("Neo4j", "neo4j", EntityType.TECHNOLOGY)]
        rels_learn = self.extractor.extract("I am learning Neo4j", entities_learn)
        self.assertEqual(len(rels_learn), 1)
        self.assertEqual(rels_learn[0].relationship_type, RelationshipType.LEARNS)

    def test_project_uses_tech_relationship(self):
        entities = [
            Entity("Jarvis", "jarvis", EntityType.PROJECT),
            Entity("Neo4j", "neo4j", EntityType.TECHNOLOGY),
        ]
        rels = self.extractor.extract("Jarvis is built with Neo4j", entities)
        # Should match Project uses Technology, plus implicit User uses Technology
        self.assertTrue(len(rels) >= 1)
        project_uses_tech = [
            r for r in rels
            if r.source_entity.entity_type == EntityType.PROJECT
            and r.target_entity.entity_type == EntityType.TECHNOLOGY
        ]
        self.assertEqual(len(project_uses_tech), 1)
        self.assertEqual(project_uses_tech[0].source_entity.name, "Jarvis")
        self.assertEqual(project_uses_tech[0].target_entity.name, "Neo4j")
        self.assertEqual(project_uses_tech[0].relationship_type, RelationshipType.USES)

    def test_implicit_inference(self):
        entities = [Entity("Firebase", "firebase", EntityType.TECHNOLOGY)]
        # Mentioning Firebase without active self/project sentence
        rels = self.extractor.extract("I think Firebase is pretty cool", entities)
        # Should infer User uses Firebase
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0].source_entity, _SELF_ENTITY)
        self.assertEqual(rels[0].target_entity.name, "Firebase")
        self.assertEqual(rels[0].relationship_type, RelationshipType.USES)


class TestGraphContextBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = GraphContextBuilder()

    def test_build_context_from_relationships(self):
        rels = [
            {
                "relationship": "WORKS_ON",
                "neighbor_name": "Jarvis",
                "outgoing": True,
            },
            {
                "relationship": "USES",
                "neighbor_name": "Python",
                "outgoing": True,
            },
        ]
        context = self.builder.build_context_from_relationships("Atharva", rels)
        self.assertIsNotNone(context)
        self.assertIn("Atharva works on Jarvis", context)
        self.assertIn("Atharva uses Python", context)

    def test_build_context_from_user_world(self):
        user_context = [
            {"relationship": "WORKS_ON", "name": "Jarvis"},
            {"relationship": "USES", "name": "Python"},
            {"relationship": "USES", "name": "Neo4j"},
        ]
        context = self.builder.build_context_from_user_world(user_context)
        self.assertIsNotNone(context)
        self.assertIn("works on: Jarvis", context)
        self.assertIn("uses: Python, Neo4j", context)


class TestNeo4jClientAndService(unittest.TestCase):
    @patch("neo4j.GraphDatabase")
    def test_client_connect_and_close(self, mock_graph_db):
        client = Neo4jClient(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="pwd",
        )
        mock_driver = MagicMock()
        mock_graph_db.driver.return_value = mock_driver

        client.connect()
        self.assertTrue(client._driver is not None)
        mock_graph_db.driver.assert_called_once_with(
            "bolt://localhost:7687",
            auth=("neo4j", "pwd"),
        )

        client.close()
        self.assertTrue(client._driver is None)
        mock_driver.close.assert_called_once()

    @patch("neo4j.GraphDatabase")
    def test_graceful_degradation_when_offline(self, mock_graph_db):
        mock_graph_db.driver.side_effect = Exception("Connection refused")
        client = Neo4jClient(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="pwd",
        )

        with self.assertRaises(Neo4jConnectionError):
            client.connect()
        self.assertFalse(client.is_connected())


if __name__ == "__main__":
    unittest.main()
