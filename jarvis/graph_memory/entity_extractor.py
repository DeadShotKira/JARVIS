"""Rule-based entity extraction for JARVIS Phase 4.

Extracts named entities from user text using a combination of:
    1. Known vocabulary matching (technologies, projects, etc.)
    2. Regex patterns for structured phrases
    3. spaCy Named Entity Recognition (optional — falls back to regex)

The ``EntityExtractor`` protocol allows Phase 4B to swap in an LLM-based
extractor without changing any downstream graph code.

Design decisions:
    - Technology vocabulary is an explicit set because rule-based NER struggles
      with tech names (``Neo4j``, ``ChromaDB``). The set is cheap to extend.
    - spaCy is imported lazily. If unavailable, extraction still works using
      regex only. This matches the project pattern in ``rag/embedder.py``.
    - Extraction is conservative — it only fires on clear signal phrases
      to avoid polluting the graph with noise from casual messages.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from jarvis.graph_memory.graph_schema import Entity, EntityType, make_canonical

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol — any extractor must satisfy this interface
# ---------------------------------------------------------------------------

class EntityExtractor(Protocol):
    """Extract named entities from text."""

    def extract(self, text: str) -> list[Entity]: ...


# ---------------------------------------------------------------------------
# Known vocabulary
# ---------------------------------------------------------------------------

KNOWN_TECHNOLOGIES: set[str] = {
    # Languages
    "Python", "Java", "JavaScript", "TypeScript", "Kotlin", "C", "C++",
    "C#", "Rust", "Go", "Ruby", "Swift", "PHP", "HTML", "CSS", "SQL",
    "Dart", "Scala", "R", "Matlab", "Bash", "Shell",
    # Frameworks & libraries
    "React", "Angular", "Vue", "Next.js", "NextJS", "Flask", "FastAPI",
    "Django", "Spring", "Express", "Node.js", "NodeJS", "PyTorch",
    "TensorFlow", "Keras", "Scikit-Learn", "Sklearn", "Pandas", "NumPy",
    "LangChain", "LlamaIndex",
    # Databases & stores
    "SQLite", "PostgreSQL", "Postgres", "MySQL", "MongoDB", "Redis",
    "Neo4j", "ChromaDB", "Faiss", "Pinecone", "Weaviate", "Milvus",
    "Elasticsearch",
    # Tools & platforms
    "Docker", "Kubernetes", "Git", "GitHub", "GitLab", "Linux", "Windows",
    "Android Studio", "VS Code", "VSCode", "IntelliJ", "Jupyter",
    "Ollama", "Firebase", "AWS", "GCP", "Azure", "Heroku", "Vercel",
    "Raspberry Pi",
    # AI / ML specific
    "spaCy", "Sentence-Transformers", "Transformers", "HuggingFace",
    "OpenAI", "Gemini", "Llama", "Qwen", "Gemma", "Mistral",
    "Stable Diffusion", "Whisper",
}

# Normalised lookup for case-insensitive matching.
_TECH_LOOKUP: dict[str, str] = {tech.lower(): tech for tech in KNOWN_TECHNOLOGIES}


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_PROJECT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:i am|i'm)\s+(?:currently\s+)?(?:building|working on|developing|creating)\s+(?P<name>.+)",
        re.IGNORECASE,
    ),
    re.compile(r"my project (?:is\s+called|is|called)\s+(?P<name>.+)", re.IGNORECASE),
    re.compile(
        r"\b(?P<name>[a-zA-Z0-9_\-]+)\s+(?:project|app|application)\b",
        re.IGNORECASE,
    ),
]

_INTEREST_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:i (?:am )?(?:interested in|enjoy|love|passionate about))\s+(?P<name>.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:my )?hobbies?\s+(?:is|are|include)\s+(?P<name>.+)",
        re.IGNORECASE,
    ),
    re.compile(r"i (?:like|prefer)\s+(?P<name>.+)", re.IGNORECASE),
]

_ORGANIZATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:i (?:study|studied|work|worked) at|i'm at|member of|part of)\s+(?P<name>.+)",
        re.IGNORECASE,
    ),
    re.compile(r"my (?:college|university|school|company) is\s+(?P<name>.+)", re.IGNORECASE),
]

_TASK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:i need to|i have to|i must|i should)\s+(?P<name>.+)", re.IGNORECASE),
    re.compile(r"remind me to\s+(?P<name>.+)", re.IGNORECASE),
    re.compile(r"todo:?\s+(?P<name>.+)", re.IGNORECASE),
]


# Words to ignore when considering potential entity extractions.
_IGNORED_STARTS = {"hello", "hi", "hey", "thanks", "thank", "yes", "no", "ok", "okay"}


# ---------------------------------------------------------------------------
# Rule-based implementation
# ---------------------------------------------------------------------------

@dataclass
class RuleBasedEntityExtractor:
    """Phase 4A entity extractor: regex + vocabulary + optional spaCy NER.

    Attributes:
        use_spacy:  If True, attempt to load spaCy for person/org detection.
                    Falls back to regex if spaCy is unavailable.
    """

    use_spacy: bool = True

    # Private — lazily loaded spaCy NLP pipeline.
    _nlp: Any = field(default=None, init=False, repr=False)
    _spacy_available: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.use_spacy:
            self._try_load_spacy()

    def extract(self, text: str) -> list[Entity]:
        """Extract entities from *text* and return a deduplicated list."""
        cleaned = _clean(text)
        if not cleaned or cleaned.split()[0].lower() in _IGNORED_STARTS:
            return []

        entities: list[Entity] = []
        entities.extend(self._extract_technologies(cleaned))
        entities.extend(self._extract_projects(cleaned))
        entities.extend(self._extract_interests(cleaned))
        entities.extend(self._extract_organizations(cleaned))
        entities.extend(self._extract_tasks(cleaned))

        if self._spacy_available:
            entities.extend(self._extract_with_spacy(cleaned))

        return _deduplicate(entities)

    # -------------------------------------------------------------------
    # Technology extraction
    # -------------------------------------------------------------------

    def _extract_technologies(self, text: str) -> list[Entity]:
        """Match known technology names anywhere in the text."""
        found: list[Entity] = []
        lowered = text.lower()
        for tech_lower, tech_display in _TECH_LOOKUP.items():
            # Word-boundary match to avoid partial hits like "go" in "going".
            # For single-character techs like "c" and "r", require exact match
            # or surrounding non-alpha to avoid false positives.
            if len(tech_lower) <= 2:
                pattern = rf"(?<![a-zA-Z]){re.escape(tech_lower)}(?![a-zA-Z])"
            else:
                pattern = rf"\b{re.escape(tech_lower)}\b"
            if re.search(pattern, lowered):
                found.append(Entity(
                    name=tech_display,
                    canonical_name=make_canonical(tech_display),
                    entity_type=EntityType.TECHNOLOGY,
                ))
        return found

    # -------------------------------------------------------------------
    # Project extraction
    # -------------------------------------------------------------------

    def _extract_projects(self, text: str) -> list[Entity]:
        for pattern in _PROJECT_PATTERNS:
            match = pattern.search(text)
            if match:
                name = _clean_fragment(match.group("name"))
                if name and len(name) > 1:
                    return [Entity(
                        name=name,
                        canonical_name=make_canonical(name),
                        entity_type=EntityType.PROJECT,
                    )]
        return []

    # -------------------------------------------------------------------
    # Interest extraction
    # -------------------------------------------------------------------

    def _extract_interests(self, text: str) -> list[Entity]:
        for pattern in _INTEREST_PATTERNS:
            match = pattern.search(text)
            if match:
                name = _clean_fragment(match.group("name"))
                if name and len(name) > 1:
                    return [Entity(
                        name=name,
                        canonical_name=make_canonical(name),
                        entity_type=EntityType.INTEREST,
                    )]
        return []

    # -------------------------------------------------------------------
    # Organization extraction
    # -------------------------------------------------------------------

    def _extract_organizations(self, text: str) -> list[Entity]:
        for pattern in _ORGANIZATION_PATTERNS:
            match = pattern.search(text)
            if match:
                name = _clean_fragment(match.group("name"))
                if name and len(name) > 1:
                    return [Entity(
                        name=name,
                        canonical_name=make_canonical(name),
                        entity_type=EntityType.ORGANIZATION,
                    )]
        return []

    # -------------------------------------------------------------------
    # Task extraction
    # -------------------------------------------------------------------

    def _extract_tasks(self, text: str) -> list[Entity]:
        for pattern in _TASK_PATTERNS:
            match = pattern.search(text)
            if match:
                name = _clean_fragment(match.group("name"))
                if name and len(name) > 2:
                    return [Entity(
                        name=name,
                        canonical_name=make_canonical(name),
                        entity_type=EntityType.TASK,
                    )]
        return []

    # -------------------------------------------------------------------
    # spaCy NER (optional)
    # -------------------------------------------------------------------

    def _try_load_spacy(self) -> None:
        """Attempt to load the spaCy English model.  Fail silently."""
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            self._spacy_available = True
            logger.info("spaCy NER loaded (en_core_web_sm).")
        except Exception:
            self._spacy_available = False
            logger.info("spaCy unavailable — using regex-only entity extraction.")

    def _extract_with_spacy(self, text: str) -> list[Entity]:
        """Use spaCy NER to find PERSON and ORG entities regex may miss."""
        if self._nlp is None:
            return []

        doc = self._nlp(text)
        entities: list[Entity] = []
        for ent in doc.ents:
            name = ent.text.strip()
            if not name or len(name) <= 1:
                continue

            if ent.label_ == "PERSON":
                entities.append(Entity(
                    name=name,
                    canonical_name=make_canonical(name),
                    entity_type=EntityType.PERSON,
                    source="spacy",
                ))
            elif ent.label_ == "ORG":
                entities.append(Entity(
                    name=name,
                    canonical_name=make_canonical(name),
                    entity_type=EntityType.ORGANIZATION,
                    source="spacy",
                ))
        return entities


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Normalise whitespace."""
    return " ".join(text.strip().split())


def _clean_fragment(text: str) -> str:
    """Strip trailing punctuation from a captured group."""
    return text.strip().strip(" .!?;:,")


def _deduplicate(entities: list[Entity]) -> list[Entity]:
    """Remove duplicate entities (same canonical_name + type)."""
    seen: dict[tuple[str, EntityType], Entity] = {}
    for entity in entities:
        key = (entity.canonical_name, entity.entity_type)
        if key not in seen:
            seen[key] = entity
    return list(seen.values())
